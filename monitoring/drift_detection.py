import json
import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("stockcast.drift")


def _extract_prediction_dataframe(prediction_log: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for item in list(prediction_log):
        if not isinstance(item, dict):
            continue

        ts = item.get("forecast_for_utc")
        value = item.get("predicted_price_usd")
        if ts is None or value is None:
            continue

        rows.append(
            {
                "timestamp": pd.to_datetime(ts, utc=True, errors="coerce"),
                "price": float(value),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.dropna(subset=["timestamp", "price"]).sort_values("timestamp")
    return df


def _extract_real_dataframe(market_df: pd.DataFrame) -> pd.DataFrame:
    if market_df is None or market_df.empty or "Close" not in market_df.columns:
        return pd.DataFrame(columns=["timestamp", "price"])

    idx = pd.to_datetime(market_df.index, utc=True, errors="coerce")
    real_df = pd.DataFrame(
        {
            "timestamp": idx,
            "price": pd.to_numeric(market_df["Close"], errors="coerce"),
        }
    )
    real_df = real_df.dropna(subset=["timestamp", "price"]).sort_values("timestamp")
    return real_df


def _build_report_with_psi() -> Any:
    """Cria Report do Evidently com DataDriftPreset priorizando PSI."""
    try:
        from evidently.report import Report
    except Exception:
        from evidently import Report

    try:
        from evidently.metric_preset import DataDriftPreset
    except Exception:
        from evidently.presets import DataDriftPreset

    try:
        from evidently.options import DataDriftOptions

        return Report(
            metrics=[DataDriftPreset()],
            options=[DataDriftOptions(num_stattest="psi", cat_stattest="psi")],
        )
    except Exception:
        try:
            return Report(metrics=[DataDriftPreset(stattest="psi")])
        except Exception:
            return Report(metrics=[DataDriftPreset()])


def _calculate_psi_fallback(
    reference_data: pd.DataFrame, current_data: pd.DataFrame, bins: int = 10
) -> float:
    """Calcula PSI manualmente quando Evidently não está operacional no ambiente."""
    ref = pd.to_numeric(reference_data["price"], errors="coerce").dropna()
    cur = pd.to_numeric(current_data["price"], errors="coerce").dropna()

    if ref.empty or cur.empty:
        return 0.0

    quantiles = [i / bins for i in range(bins + 1)]
    cut_points = ref.quantile(quantiles).drop_duplicates().to_numpy()
    if len(cut_points) < 3:
        return 0.0

    cut_points[0] = -np.inf
    cut_points[-1] = np.inf

    ref_bins = pd.cut(ref, bins=cut_points, include_lowest=True)
    cur_bins = pd.cut(cur, bins=cut_points, include_lowest=True)

    ref_pct = ref_bins.value_counts(normalize=True, sort=False)
    cur_pct = cur_bins.value_counts(normalize=True, sort=False)

    epsilon = 1e-6
    ref_pct = ref_pct.reindex(ref_pct.index.union(cur_pct.index), fill_value=0.0) + epsilon
    cur_pct = cur_pct.reindex(ref_pct.index, fill_value=0.0) + epsilon

    psi = ((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)).sum()
    return float(psi)


def _extract_psi(report_dict: dict[str, Any]) -> float:
    max_psi = 0.0

    for metric in report_dict.get("metrics", []):
        result = metric.get("result", {})
        drift_by_columns = result.get("drift_by_columns", {})

        if isinstance(drift_by_columns, dict):
            for _, data in drift_by_columns.items():
                stattest_name = str(data.get("stattest_name", "")).lower()
                drift_score = data.get("drift_score")
                if drift_score is None:
                    continue
                if "psi" in stattest_name:
                    max_psi = max(max_psi, float(drift_score))

    return max_psi


async def detect_data_drift(
    ticker: str = "BTC-USD",
    download_fn: Callable[[str], tuple[pd.DataFrame, str]] | None = None,
    prediction_log: Any = None,
) -> dict[str, Any]:
    """Executa detecção de data drift comparando predições históricas e dados reais recém-baixados.

    Fonte de dados e histórico:
      - download_with_retry(ticker) de src.app
      - prediction_log (collections.deque) de src.app
    """
    if download_fn is None or prediction_log is None:
        from src import app as app_module

        if download_fn is None:
            download_fn = app_module.download_with_retry
        if prediction_log is None:
            prediction_log = app_module.prediction_log

    market_df, data_source = download_fn(ticker)
    prediction_df = _extract_prediction_dataframe(prediction_log)
    real_df = _extract_real_dataframe(market_df)

    if prediction_df.empty or real_df.empty:
        payload = {
            "event": "data_drift_skipped",
            "reason": "missing_data",
            "prediction_rows": int(len(prediction_df)),
            "real_rows": int(len(real_df)),
            "ticker": ticker,
        }
        logger.warning(json.dumps(payload, ensure_ascii=False))
        return {"status": "skipped", **payload}

    merged = pd.merge(
        prediction_df, real_df, on="timestamp", how="inner", suffixes=("_pred", "_real")
    )

    if merged.empty:
        payload = {
            "event": "data_drift_skipped",
            "reason": "no_timestamp_overlap",
            "prediction_rows": int(len(prediction_df)),
            "real_rows": int(len(real_df)),
            "ticker": ticker,
        }
        logger.warning(json.dumps(payload, ensure_ascii=False))
        return {"status": "skipped", **payload}

    reference_data = merged[["price_pred"]].rename(columns={"price_pred": "price"})
    current_data = merged[["price_real"]].rename(columns={"price_real": "price"})

    used_evidently = True
    try:
        report = _build_report_with_psi()
        report.run(reference_data=reference_data, current_data=current_data)
        report_dict = report.as_dict()
        psi = _extract_psi(report_dict)
    except Exception as exc:
        used_evidently = False
        psi = _calculate_psi_fallback(reference_data, current_data)
        logger.warning(
            json.dumps(
                {
                    "event": "data_drift_evidently_fallback",
                    "reason": str(exc),
                    "psi_fallback": float(psi),
                },
                ensure_ascii=False,
            )
        )

    payload = {
        "event": "data_drift_evaluated",
        "ticker": ticker,
        "data_source": data_source,
        "rows_compared": int(len(merged)),
        "psi": float(psi),
        "used_evidently": used_evidently,
        "threshold_warning": 0.1,
        "threshold_retrain": 0.2,
    }

    if psi > 0.2:
        logger.error(
            json.dumps({**payload, "action": "simulate_retrain_trigger"}, ensure_ascii=False)
        )
    elif psi > 0.1:
        logger.warning(json.dumps(payload, ensure_ascii=False))
    else:
        logger.info(json.dumps(payload, ensure_ascii=False))

    return {
        "status": "ok",
        "psi": float(psi),
        "rows_compared": int(len(merged)),
        "data_source": data_source,
    }
