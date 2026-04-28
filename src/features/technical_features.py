from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "log_return",
    "rsi",
    "macd_signal",
    "bb_pct_b",
    "sma_ratio",
    "vol_ratio",
]


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def compute_macd_signal(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.Series:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    normalized = (macd_line - signal_line) / series.replace(0, np.nan)
    return normalized.fillna(0.0)


def compute_bollinger_pct_b(series: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.Series:
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    band_width = (upper - lower).replace(0, np.nan)
    pct_b = (series - lower) / band_width
    return pct_b.fillna(0.5).clip(0.0, 1.0)


def compute_sma_ratio(series: pd.Series, short: int = 7, long: int = 21) -> pd.Series:
    sma_short = series.rolling(window=short).mean()
    sma_long = series.rolling(window=long).mean()
    ratio = (sma_short / sma_long.replace(0, np.nan)) - 1.0
    return ratio.fillna(0.0)


def compute_volume_ratio(volume: pd.Series, period: int = 24) -> pd.Series:
    vol_sma = volume.rolling(window=period).mean()
    ratio = volume / vol_sma.replace(0, np.nan)
    return ratio.fillna(1.0).clip(0.0, 10.0)


def build_feature_matrix(data: pd.DataFrame) -> pd.DataFrame:
    close = data["Close"]
    volume = data["Volume"]

    log_return = np.log(close).diff()
    rsi = compute_rsi(close, 14) / 100.0
    macd_sig = compute_macd_signal(close)
    bb_pct = compute_bollinger_pct_b(close)
    sma_ratio = compute_sma_ratio(close)
    vol_ratio = compute_volume_ratio(volume)

    features = pd.DataFrame(
        {
            "log_return": log_return,
            "rsi": rsi,
            "macd_signal": macd_sig,
            "bb_pct_b": bb_pct,
            "sma_ratio": sma_ratio,
            "vol_ratio": vol_ratio,
        },
        index=data.index,
    )

    return features.dropna()