import numpy as np
import pandas as pd
import pytest

from src import app as app_module


def test_remove_incomplete_hour_candle_drops_open_hour():
    now_hour = pd.Timestamp.utcnow().floor("h")
    index = pd.DatetimeIndex([now_hour - pd.Timedelta(hours=1), now_hour])
    series = pd.Series([100.0, 101.0], index=index)

    cleaned = app_module.remove_incomplete_hour_candle(series)
    assert len(cleaned) == 1


def test_remove_incomplete_hour_candle_keeps_closed_hours():
    now_hour = pd.Timestamp.utcnow().floor("h")
    index = pd.DatetimeIndex([now_hour - pd.Timedelta(hours=3), now_hour - pd.Timedelta(hours=2)])
    series = pd.Series([100.0, 101.0], index=index)

    cleaned = app_module.remove_incomplete_hour_candle(series)
    assert len(cleaned) == 2


def test_indicator_functions_ranges(synthetic_yfinance_df):
    close = synthetic_yfinance_df["Close"]
    volume = synthetic_yfinance_df["Volume"]

    rsi = app_module._compute_rsi(close) / 100.0
    macd = app_module._compute_macd_signal(close)
    bb = app_module._compute_bollinger_pct_b(close)
    sma = app_module._compute_sma_ratio(close)
    vol = app_module._compute_volume_ratio(volume)

    assert rsi.between(0, 1).all()
    assert bb.between(0, 1).all()
    assert vol.between(0, 10).all()
    assert np.isfinite(macd).all()
    assert np.isfinite(sma).all()


def test_cache_helpers_roundtrip(synthetic_yfinance_df):
    app_module.set_cached_market_data("BTC-USD", synthetic_yfinance_df, "yfinance")
    cached = app_module.get_cached_market_data("BTC-USD")
    source = app_module.get_cached_source("BTC-USD")

    assert cached is not None
    assert len(cached) == len(synthetic_yfinance_df)
    assert source == "yfinance"


def test_timestamp_helpers():
    ts = pd.Timestamp("2026-01-01T10:00:00Z")
    utc_iso = app_module.timestamp_to_utc_iso(ts)
    brt_iso = app_module.timestamp_to_brt_iso(ts)

    assert utc_iso.endswith("+00:00")
    assert "-03:00" in brt_iso or "-02:00" in brt_iso


def test_estimate_uncertainty_with_rmse_and_mape():
    metadata = {"metrics": {"rmse_price": 100.0, "mape_price": 2.5}}
    err_pct, ci = app_module.estimate_uncertainty(10000.0, metadata)

    assert err_pct == 2.5
    assert ci is not None
    assert ci.low_usd < ci.high_usd


def test_estimate_uncertainty_without_metrics_returns_none_ci():
    err_pct, ci = app_module.estimate_uncertainty(10000.0, {})
    assert err_pct is None
    assert ci is None


def test_is_production_environment_true_when_app_env_is_production(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")

    assert app_module.is_production_environment() is True


def test_validate_bedrock_configuration_for_startup_allows_non_production(monkeypatch):
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.delenv("ENV", raising=False)
    monkeypatch.delenv("STAGE", raising=False)
    monkeypatch.delenv("DEPLOY_ENV", raising=False)
    monkeypatch.setenv("AWS_REGION", "local-test-1")

    app_module.validate_bedrock_configuration_for_startup()


def test_validate_bedrock_configuration_for_startup_rejects_placeholder_in_production(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("AWS_REGION", "your-aws-region")

    with pytest.raises(RuntimeError, match="inválida para produção"):
        app_module.validate_bedrock_configuration_for_startup()


def test_validate_bedrock_configuration_for_startup_rejects_invalid_format_in_production(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("AWS_REGION", "insecure-region")

    with pytest.raises(RuntimeError, match="formato inválido"):
        app_module.validate_bedrock_configuration_for_startup()


def test_validate_bedrock_configuration_for_startup_accepts_valid_format_in_production(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("BEDROCK_GUARDRAIL_ID", "gr-123")
    monkeypatch.setenv("BEDROCK_GUARDRAIL_VERSION", "1")

    app_module.validate_bedrock_configuration_for_startup()


def test_validate_bedrock_configuration_for_startup_requires_guardrail_in_production(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.delenv("BEDROCK_GUARDRAIL_ID", raising=False)
    monkeypatch.delenv("BEDROCK_GUARDRAIL_VERSION", raising=False)

    with pytest.raises(RuntimeError, match="Guardrails"):
        app_module.validate_bedrock_configuration_for_startup()
