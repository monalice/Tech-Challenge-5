import tensorflow as tf  # noqa: F401
import yfinance as yf  # noqa: F401
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_yfinance_df() -> pd.DataFrame:
    periods = 200
    index = pd.date_range(end=pd.Timestamp.utcnow(), periods=periods, freq="h", tz="UTC")

    base_price = 100_000.0
    trend = np.linspace(0, 1_500, periods)
    seasonality = 600 * np.sin(np.linspace(0, 8 * np.pi, periods))
    noise = np.random.default_rng(42).normal(0, 80, periods)

    close = base_price + trend + seasonality + noise
    high = close + np.abs(np.random.default_rng(7).normal(80, 20, periods))
    low = close - np.abs(np.random.default_rng(9).normal(80, 20, periods))
    open_ = close + np.random.default_rng(11).normal(0, 30, periods)
    volume = np.abs(np.random.default_rng(13).normal(250, 40, periods))

    df = pd.DataFrame(
        {
            "Open": open_.astype(float),
            "High": high.astype(float),
            "Low": low.astype(float),
            "Close": close.astype(float),
            "Volume": volume.astype(float),
        },
        index=index,
    )
    df.index.name = "Datetime"
    return df
