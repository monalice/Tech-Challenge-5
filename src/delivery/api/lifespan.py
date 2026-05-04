import json
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI

from src.adapters.ml.model_loader import load_trained_model
from src.adapters.ml.s3_model_manager import S3ModelManager
from src.agent.llm_config import validate_bedrock_configuration_for_startup
from src.delivery.api.dependencies import _DownloadWithRetryPort
from src.domain.constants import LOOKBACK, MODEL_PATH, SCALER_PATH, SUPPORTED_TICKER
from src.domain.inference import InferenceService
from src.domain.ports import LoadedArtifacts

logger = logging.getLogger("stockcast")

SCALER_RETURN_PATH = "models/scaler_btc_return.gz"
MODEL_META_PATH = "models/model_metadata_btc.json"

# Configuração de S3 para modelos (via variável de ambiente)
S3_MODELS_BUCKET = os.getenv("S3_MODELS_BUCKET", "").strip()
s3_manager = S3ModelManager(bucket_name=S3_MODELS_BUCKET if S3_MODELS_BUCKET else None)

# Prefixo S3 do modelo em produção (champion)
_S3_CHAMPION_PREFIX = "champion"
STRICT_ARTIFACT_STARTUP = os.getenv("STRICT_ARTIFACT_STARTUP", "false").strip().lower() in {
    "1",
    "true",
    "yes",
}


def _try_load_optional_scaler(scaler_path: str) -> bool:
    """Verifica se o scaler opcional existe (local ou S3)."""
    if S3_MODELS_BUCKET and s3_manager.s3_enabled:
        try:
            s3_manager.load_joblib(f"{_S3_CHAMPION_PREFIX}/{scaler_path.split('/')[-1]}")
            return True
        except FileNotFoundError:
            return False
    # Fallback local
    return os.path.exists(scaler_path)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    validate_bedrock_configuration_for_startup()
    logger.info("Carregando modelo LSTM Hourly e scaler...")
    try:
        if S3_MODELS_BUCKET and s3_manager.s3_enabled:
            _model = s3_manager.load_model(f"{_S3_CHAMPION_PREFIX}/lstm_btc_hourly.keras")
            _scaler = s3_manager.load_joblib(f"{_S3_CHAMPION_PREFIX}/{SCALER_PATH.split('/')[-1]}")
        else:
            _model = load_trained_model(MODEL_PATH)
            _scaler = joblib.load(SCALER_PATH)
        _scaler_return = (
            (
                s3_manager.load_joblib(f"{_S3_CHAMPION_PREFIX}/{SCALER_RETURN_PATH.split('/')[-1]}")
                if S3_MODELS_BUCKET and s3_manager.s3_enabled
                else joblib.load(SCALER_RETURN_PATH)
            )
            if _try_load_optional_scaler(SCALER_RETURN_PATH)
            else None
        )

        try:
            if S3_MODELS_BUCKET and s3_manager.s3_enabled:
                meta_key = s3_manager._s3_key(
                    f"{_S3_CHAMPION_PREFIX}/{MODEL_META_PATH.split('/')[-1]}"
                )
                meta_response = s3_manager.s3_client.get_object(
                    Bucket=S3_MODELS_BUCKET,
                    Key=meta_key,
                )
                meta_content = meta_response["Body"].read().decode("utf-8")
                _metadata = json.loads(meta_content)
            else:
                with open(MODEL_META_PATH, encoding="utf-8") as meta_file:
                    _metadata = json.load(meta_file)
        except (FileNotFoundError, ValueError, TypeError):
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
        logger.exception("Falha ao carregar artefatos na inicialização: %s", e)
        if STRICT_ARTIFACT_STARTUP:
            raise RuntimeError(f"Falha crítica ao carregar artefatos do modelo: {e}") from e
        logger.warning(
            "API iniciada em modo degradado (sem artefatos). "
            "Defina STRICT_ARTIFACT_STARTUP=true para falhar o startup."
        )
    yield
    app.state.artifacts = None
    app.state.service = None
    app.state.agent_llm = None
    logger.info("Artefatos descarregados. API encerrada.")
