from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.technical_features import (
    FEATURE_COLUMNS,
    build_feature_matrix,
    compute_bollinger_pct_b,
    compute_macd_signal,
    compute_rsi,
    compute_sma_ratio,
    compute_volume_ratio,
)


def _ohlcv_df(periods: int = 120) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=periods, freq="h", tz="UTC")
    close = np.linspace(100_000.0, 101_500.0, periods)
    high = close + 80.0
    low = close - 80.0
    volume = np.linspace(100.0, 500.0, periods)
    return pd.DataFrame(
        {
            "Close": close,
            "High": high,
            "Low": low,
            "Volume": volume,
        },
        index=index,
    )


def test_feature_build_has_expected_columns_and_no_nans() -> None:
    df = _ohlcv_df()

    features = build_feature_matrix(df)

    assert list(features.columns) == FEATURE_COLUMNS
    assert not features.empty
    assert not features.isna().any().any()


def test_indicator_ranges_and_finiteness() -> None:
    df = _ohlcv_df()

    rsi = compute_rsi(df["Close"]) / 100.0
    macd = compute_macd_signal(df["Close"])
    bb_pct = compute_bollinger_pct_b(df["Close"])
    sma_ratio = compute_sma_ratio(df["Close"])
    vol_ratio = compute_volume_ratio(df["Volume"])

    assert rsi.dropna().between(0.0, 1.0).all()
    assert bb_pct.dropna().between(0.0, 1.0).all()
    assert vol_ratio.dropna().between(0.0, 10.0).all()
    assert np.isfinite(macd).all()
    assert np.isfinite(sma_ratio).all()
