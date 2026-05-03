"""Utilitários de tempo e tratamento de candles compartilhados entre módulos.

Fonte única de verdade para conversões de timestamp e remoção do candle
horário em formação, eliminando a duplicação identificada na auditoria (M3).
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd

from src.domain.constants import BRASILIA_TZ

# ---------------------------------------------------------------------------
# Remoção de candle parcial
# ---------------------------------------------------------------------------


def remove_incomplete_hour_candle(series: pd.Series) -> pd.Series:
    """Remove o candle horário parcial (em formação) de uma série temporal.

    Compara o último timestamp da série com a hora atual UTC truncada. Se o
    último candle corresponder à hora corrente (ainda não fechada), ele é
    descartado para evitar ruído na inferência do modelo.

    Args:
        series: Série temporal de preços indexada por timestamps (tz-aware ou naive).

    Returns:
        Série sem o último elemento se ele corresponder à hora em formação;
        caso contrário, a série original sem modificação.
    """
    if len(series) < 2:
        return series

    last_ts = pd.Timestamp(series.index[-1])
    now_utc = pd.Timestamp.utcnow()
    now_ref = (
        now_utc.tz_localize(None) if last_ts.tzinfo is None else now_utc.tz_convert(last_ts.tz)
    )
    if last_ts >= now_ref.floor("h"):
        return series.iloc[:-1]
    return series


def remove_incomplete_hour_candle_df(df: pd.DataFrame) -> pd.DataFrame:
    """Remove a última linha do DataFrame se ela corresponder ao candle em formação.

    Args:
        df: DataFrame com índice DatetimeIndex. Deve ter pelo menos 2 linhas.

    Returns:
        DataFrame sem a última linha quando ela representa a hora corrente em
        formação; caso contrário, o DataFrame original sem modificação.
    """
    if len(df) < 2:
        return df

    last_ts = pd.Timestamp(df.index[-1])
    now_utc = pd.Timestamp.utcnow()
    now_ref = (
        now_utc.tz_localize(None) if last_ts.tzinfo is None else now_utc.tz_convert(last_ts.tz)
    )
    if last_ts >= now_ref.floor("h"):
        return df.iloc[:-1]
    return df


# ---------------------------------------------------------------------------
# Conversão de timestamps
# ---------------------------------------------------------------------------


def timestamp_to_utc_iso(ts: pd.Timestamp) -> str:
    """Converte um timestamp para string ISO-8601 em UTC.

    Args:
        ts: Timestamp pandas, tz-aware ou naive (assumido UTC se naive).

    Returns:
        String ISO-8601 com offset UTC (ex: ``"2026-04-18T14:00:00+00:00"``).
    """
    ts = pd.Timestamp(ts)
    ts_utc = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return str(ts_utc.isoformat())


def timestamp_to_brt_iso(ts: pd.Timestamp) -> str:
    """Converte um timestamp para string ISO-8601 no horário de Brasília (BRT/BRST).

    Args:
        ts: Timestamp pandas, tz-aware ou naive (assumido UTC se naive).

    Returns:
        String ISO-8601 com offset de Brasília (ex: ``"2026-04-18T11:00:00-03:00"``).
    """
    ts = pd.Timestamp(ts)
    ts_utc = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return str(ts_utc.tz_convert(ZoneInfo(BRASILIA_TZ)).isoformat())
