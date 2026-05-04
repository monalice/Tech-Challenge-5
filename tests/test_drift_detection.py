import asyncio
from collections import deque

import pandas as pd

from src.domain.drift import detection as dd


def _prediction_log(periods: int = 20):
    index = pd.date_range("2026-01-01", periods=periods, freq="h", tz="UTC")
    return deque(
        [
            {
                "forecast_for_utc": ts.isoformat(),
                "predicted_price_usd": float(100 + i),
            }
            for i, ts in enumerate(index)
        ],
        maxlen=100,
    )


def _real_df(periods: int = 20, start: str = "2026-01-01", base: float = 100.0):
    index = pd.date_range(start, periods=periods, freq="h", tz="UTC")
    return pd.DataFrame({"Close": [base + i for i in range(periods)]}, index=index)


def test_extract_prediction_dataframe_filters_invalid_items():
    log = deque(
        [
            {"forecast_for_utc": "2026-01-01T00:00:00+00:00", "predicted_price_usd": 100.0},
            {"forecast_for_utc": None, "predicted_price_usd": 101.0},
            {"foo": "bar"},
            "invalid",
        ]
    )
    df = dd._extract_prediction_dataframe(log)
    assert len(df) == 1
    assert list(df.columns) == ["timestamp", "price"]


def test_extract_real_dataframe_handles_missing_close():
    empty_df = dd._extract_real_dataframe(pd.DataFrame({"Open": [1, 2]}))
    assert empty_df.empty


def test_calculate_psi_fallback_returns_positive_for_different_distributions():
    ref = pd.DataFrame({"price": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]})
    cur = pd.DataFrame({"price": [200, 201, 202, 203, 204, 205, 206, 207, 208, 209]})
    psi = dd._calculate_psi_fallback(ref, cur)
    assert psi > 0.1


def test_detect_data_drift_skips_when_missing_data(caplog):
    caplog.set_level("WARNING")
    result = asyncio.run(
        dd.detect_data_drift(
            download_fn=lambda ticker: (pd.DataFrame(), "mock"),
            prediction_log=deque(),
        )
    )
    assert result["status"] == "skipped"
    assert "missing_data" in caplog.text


def test_detect_data_drift_skips_when_no_overlap(caplog):
    caplog.set_level("WARNING")
    pred_log = _prediction_log(10)
    real = _real_df(10, start="2027-01-01")

    result = asyncio.run(
        dd.detect_data_drift(
            download_fn=lambda ticker: (real, "mock"),
            prediction_log=pred_log,
        )
    )
    assert result["status"] == "skipped"
    assert "no_timestamp_overlap" in caplog.text


def test_detect_data_drift_logs_warning_for_psi_above_01(monkeypatch, caplog):
    class DummyReport:
        def run(self, reference_data, current_data):
            return None

        def as_dict(self):
            return {
                "metrics": [
                    {
                        "result": {
                            "drift_by_columns": {
                                "price": {"stattest_name": "psi", "drift_score": 0.15}
                            }
                        }
                    }
                ]
            }

    monkeypatch.setattr(dd, "_build_report_with_psi", lambda: DummyReport())

    caplog.set_level("WARNING")
    result = asyncio.run(
        dd.detect_data_drift(
            download_fn=lambda ticker: (_real_df(20), "mock"),
            prediction_log=_prediction_log(20),
        )
    )

    assert result["status"] == "ok"
    assert result["psi"] == 0.15
    assert "data_drift_evaluated" in caplog.text


def test_detect_data_drift_logs_error_for_psi_above_02(monkeypatch, caplog):
    class DummyReport:
        def run(self, reference_data, current_data):
            return None

        def as_dict(self):
            return {
                "metrics": [
                    {
                        "result": {
                            "drift_by_columns": {
                                "price": {"stattest_name": "psi", "drift_score": 0.25}
                            }
                        }
                    }
                ]
            }

    monkeypatch.setattr(dd, "_build_report_with_psi", lambda: DummyReport())

    caplog.set_level("ERROR")
    result = asyncio.run(
        dd.detect_data_drift(
            download_fn=lambda ticker: (_real_df(20), "mock"),
            prediction_log=_prediction_log(20),
        )
    )

    assert result["status"] == "ok"
    assert result["psi"] == 0.25
    assert "simulate_retrain_trigger" in caplog.text


def test_detect_data_drift_uses_fallback_when_evidently_errors(monkeypatch, caplog):
    monkeypatch.setattr(
        dd,
        "_build_report_with_psi",
        lambda: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    caplog.set_level("WARNING")
    result = asyncio.run(
        dd.detect_data_drift(
            download_fn=lambda ticker: (_real_df(20, base=200), "mock"),
            prediction_log=_prediction_log(20),
        )
    )

    assert result["status"] == "ok"
    assert "data_drift_evidently_fallback" in caplog.text
    assert result["psi"] >= 0.0
