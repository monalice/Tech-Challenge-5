from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import HTTPException
import numpy as np
import pandas as pd
import psutil

from src.domain.constants import LOOKBACK, SUPPORTED_TICKER
from src.domain.ports import LoadedArtifacts
from src.domain.time_utils import (
    remove_incomplete_hour_candle,
    timestamp_to_brt_iso,
    timestamp_to_utc_iso,
)


def perform_health_checks(
    artifacts: LoadedArtifacts | None,
    *,
    download_market_data: Callable[[str], tuple[pd.DataFrame, str]],
    metric_cpu: Any,
    metric_memory: Any,
) -> dict[str, Any]:
    """Executa checks de saúde da API e retorna o resultado consolidado."""
    model: Any | None = artifacts.model if artifacts else None
    scaler = artifacts.scaler if artifacts else None

    artifacts_ready = model is not None and scaler is not None
    model_usable = False
    market_data_accessible = False
    active_source = None
    last_market_timestamp_utc = None
    last_market_timestamp_brt = None
    issues = []

    if artifacts_ready:
        try:
            metadata_check = artifacts.metadata if artifacts else {}
            n_features = metadata_check.get("n_features", 1)
            sample_input = np.zeros((1, LOOKBACK, n_features), dtype=np.float32)
            if model is None:
                raise ValueError("Modelo indisponível")
            prediction = model.predict(sample_input, verbose=0)
            if prediction is None or len(prediction) == 0:
                raise ValueError("Predição vazia do modelo")
            model_usable = True
        except Exception:
            issues.append("Modelo carregado, mas não respondeu a inferência de saúde")
    else:
        issues.append("Artefatos de modelo/scaler não carregados")

    try:
        df, active_source = download_market_data(SUPPORTED_TICKER)
        close_series = df["Close"].dropna()
        close_series = remove_incomplete_hour_candle(close_series)

        if len(close_series) == 0:
            raise ValueError("Sem candles válidos")

        market_data_accessible = True
        last_market_ts = pd.Timestamp(close_series.index[-1])
        last_market_timestamp_utc = timestamp_to_utc_iso(last_market_ts)
        last_market_timestamp_brt = timestamp_to_brt_iso(last_market_ts)
    except HTTPException:
        issues.append("Dados de mercado indisponíveis em todas as fontes")
    except Exception:
        issues.append("Dados de mercado indisponíveis no momento")

    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory().percent
    metric_cpu.set(cpu)
    metric_memory.set(mem)

    healthy = artifacts_ready and model_usable and market_data_accessible
    return {
        "status": "healthy" if healthy else "degraded",
        "artifacts_ready": artifacts_ready,
        "model_usable": model_usable,
        "market_data_accessible": market_data_accessible,
        "data_source": active_source,
        "last_market_timestamp_utc": last_market_timestamp_utc,
        "last_market_timestamp_brt": last_market_timestamp_brt,
        "cpu_usage": cpu,
        "memory_usage": mem,
        "details": None if healthy else " | ".join(issues),
    }
