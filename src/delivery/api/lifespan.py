import json
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import joblib
from fastapi import FastAPI

from src.adapters.ml.model_loader import load_trained_model
from src.agent.llm_config import validate_bedrock_configuration_for_startup
from src.delivery.api.dependencies import _DownloadWithRetryPort
from src.domain.constants import LOOKBACK, MODEL_PATH, SCALER_PATH, SUPPORTED_TICKER
from src.domain.inference import InferenceService
from src.domain.ports import LoadedArtifacts

logger = logging.getLogger("stockcast")

SCALER_RETURN_PATH = "models/scaler_btc_return.gz"
MODEL_META_PATH = "models/model_metadata_btc.json"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    validate_bedrock_configuration_for_startup()
    logger.info("Carregando modelo LSTM Hourly e scaler...")
    try:
        _model = load_trained_model(MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)
        _scaler_return = (
            joblib.load(SCALER_RETURN_PATH) if os.path.exists(SCALER_RETURN_PATH) else None
        )

        try:
            with open(MODEL_META_PATH, encoding="utf-8") as meta_file:
                _metadata: dict[str, Any] = json.load(meta_file)
        except FileNotFoundError:
            _metadata = {
                "target": "log_return",
                "lookback": LOOKBACK,
                "ticker": SUPPORTED_TICKER,
            }

        artifacts = LoadedArtifacts(
            model=_model,
            scaler=_scaler,
            scaler_return=_scaler_return,
            metadata=_metadata,
        )
        app.state.artifacts = artifacts
        app.state.service = InferenceService(artifacts, _DownloadWithRetryPort())
        try:
            # Import lazy para evitar custo de inicialização do stack LangChain fora do uso.
            from src.agent.react_agent import create_agent_llm  # noqa: PLC0415

            app.state.agent_llm = create_agent_llm()
        except OSError as exc:
            app.state.agent_llm = None
            logger.warning("LLM do agente indisponível durante startup: %s", exc)
        logger.info("Artefatos carregados com sucesso.")
    except Exception as e:
        app.state.artifacts = None
        app.state.service = None
        app.state.agent_llm = None
        raise RuntimeError(f"Falha crítica ao carregar artefatos do modelo: {e}") from e
    yield
    app.state.artifacts = None
    app.state.service = None
    app.state.agent_llm = None
    logger.info("Artefatos descarregados. API encerrada.")
