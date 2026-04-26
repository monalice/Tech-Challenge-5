import numpy as np
import pandas as pd
import pandera as pa
from pandera import Check

from src import app


def _raw_schema() -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        {
            "Open": pa.Column(float, nullable=False),
            "High": pa.Column(float, nullable=False),
            "Low": pa.Column(float, nullable=False),
            "Close": pa.Column(float, nullable=False),
            "Volume": pa.Column(float, nullable=False, checks=Check.ge(0)),
        },
        strict=True,
    )


def _features_schema() -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        {
            "log_return": pa.Column(float, nullable=False),
            "rsi": pa.Column(float, nullable=False, checks=[Check.ge(0), Check.le(1)]),
            "macd_signal": pa.Column(float, nullable=False),
            "bb_pct_b": pa.Column(float, nullable=False, checks=[Check.ge(0), Check.le(1)]),
            "sma_ratio": pa.Column(float, nullable=False),
            "vol_ratio": pa.Column(float, nullable=False, checks=[Check.ge(0), Check.le(10)]),
        },
        strict=True,
    )


def test_validate_raw_market_dataframe_schema(synthetic_yfinance_df: pd.DataFrame):
    validated = _raw_schema().validate(synthetic_yfinance_df)
    assert not validated.empty
    assert list(validated.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_validate_indicator_dataframe_schema(synthetic_yfinance_df: pd.DataFrame):
    raw_df = _raw_schema().validate(synthetic_yfinance_df)

    close_col = raw_df["Close"]
    log_return = np.log(close_col.astype(float)).diff()

    features_df = pd.DataFrame(
        {
            "log_return": log_return,
            "rsi": app._compute_rsi(close_col, 14) / 100.0,
            "macd_signal": app._compute_macd_signal(close_col),
            "bb_pct_b": app._compute_bollinger_pct_b(close_col),
            "sma_ratio": app._compute_sma_ratio(close_col),
            "vol_ratio": app._compute_volume_ratio(raw_df["Volume"]),
        },
        index=raw_df.index,
    )

    features_df = features_df.dropna()

    validated = _features_schema().validate(features_df)
    assert not validated.empty
    assert list(validated.columns) == [
        "log_return",
        "rsi",
        "macd_signal",
        "bb_pct_b",
        "sma_ratio",
        "vol_ratio",
    ]
