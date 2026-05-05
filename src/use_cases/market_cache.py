from __future__ import annotations

import time
from typing import Any

import pandas as pd

CACHE_TTL_SECONDS = 30
_cache: dict[str, dict[str, Any]] = {}


def get_cached_market_data(ticker: str) -> pd.DataFrame | None:
    """Recupera dados de mercado do cache em memória se ainda estiverem válidos."""
    cache_entry = _cache.get(ticker)
    if not cache_entry:
        return None

    age_seconds = time.time() - cache_entry["cached_at"]
    if age_seconds > CACHE_TTL_SECONDS:
        return None

    return cache_entry["data"].copy()


def set_cached_market_data(ticker: str, data: pd.DataFrame, source: str) -> None:
    """Armazena dados de mercado no cache em memória com timestamp de inserção."""
    _cache[ticker] = {"cached_at": time.time(), "data": data.copy(), "source": source}


def get_cached_source(ticker: str) -> str:
    """Retorna a fonte de dados registrada no cache para o ticker informado."""
    entry = _cache.get(ticker)
    if not entry:
        return "unknown"
    return str(entry.get("source", "unknown"))
