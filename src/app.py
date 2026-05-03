import logging
import os

import joblib  # noqa: F401 — exposto para monkeypatch em tests: app_module.joblib
import uvicorn
from fastapi import FastAPI

# --- Backward-compat re-exports (acessados via `from src import app as app_module`) ---
from src.adapters.ml.model_loader import load_trained_model  # noqa: F401
from src.agent.llm_config import (  # noqa: F401
    is_production_environment,
    validate_bedrock_configuration_for_startup,
)
from src.delivery.api.dependencies import (  # noqa: F401
    download_with_retry,
    estimate_uncertainty,
    prediction_log,
)
from src.delivery.api.lifespan import lifespan
from src.delivery.api.routers import admin, chat, health, predict
from src.domain.features.technical_features import (  # noqa: F401
    compute_bollinger_pct_b as _compute_bollinger_pct_b,
)
from src.domain.features.technical_features import (
    compute_macd_signal as _compute_macd_signal,
)
from src.domain.features.technical_features import (
    compute_rsi as _compute_rsi,
)
from src.domain.features.technical_features import (
    compute_sma_ratio as _compute_sma_ratio,
)
from src.domain.features.technical_features import (
    compute_volume_ratio as _compute_volume_ratio,
)
from src.domain.time_utils import (  # noqa: F401
    remove_incomplete_hour_candle,
    timestamp_to_brt_iso,
    timestamp_to_utc_iso,
)
from src.use_cases.market_cache import (  # noqa: F401
    get_cached_market_data,
    get_cached_source,
    set_cached_market_data,
)

__all__ = [
    "app",
    "download_with_retry",
    "estimate_uncertainty",
    "prediction_log",
    "load_trained_model",
    "is_production_environment",
    "validate_bedrock_configuration_for_startup",
    "_compute_bollinger_pct_b",
    "_compute_macd_signal",
    "_compute_rsi",
    "_compute_sma_ratio",
    "_compute_volume_ratio",
    "remove_incomplete_hour_candle",
    "timestamp_to_brt_iso",
    "timestamp_to_utc_iso",
    "get_cached_market_data",
    "get_cached_source",
    "set_cached_market_data",
]

# --- Logging estruturado ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)

app = FastAPI(title="Bitcoin Hourly Forecaster", version="3.0.0", lifespan=lifespan)
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(chat.router)
app.include_router(admin.router)

if __name__ == "__main__":
    uvicorn_host = os.getenv("UVICORN_HOST", "127.0.0.1")
    uvicorn_port = int(os.getenv("UVICORN_PORT", "8000"))
    uvicorn.run(app, host=uvicorn_host, port=uvicorn_port)
