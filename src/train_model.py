import json
import logging
import os
import subprocess
import tempfile
import time
from typing import Any

import joblib
import mlflow
import mlflow.keras
import numpy as np
import pandas as pd
import pandera.pandas as pa
import requests
import yaml  # type: ignore[import-untyped]
from numpy.typing import NDArray

# IMPORTANTE (Windows): tensorflow e yfinance devem ser importados antes de pandas
# para evitar conflito de DLL que causa crash (exit code -1073741819)
import tensorflow as tf
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from src.domain.constants import (
    BINANCE_API_URL,
    BINANCE_SYMBOL,
    BINANCE_TIMEOUT_SECONDS,
    LOOKBACK,
    MODEL_META_PATH,
    MODEL_PATH,
    SCALER_PATH,
    SCALER_RETURN_PATH,
    TICKER,
)
from src.domain.features.technical_features import (
    FEATURE_COLUMNS,
    build_feature_matrix as _build_feature_matrix,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Configurações
PERIOD = "730d"
INTERVAL = "1h"

BATCH_SIZE = 64
EPOCHS = 100
TEST_SIZE_PCT = 0.2
VAL_SIZE_PCT = 0.1
WALK_FORWARD_SPLITS = 3
WALK_FORWARD_EPOCHS = 20
RANDOM_SEED = 42
EPSILON = 1e-8
DOWNLOAD_MAX_RETRIES = 5
DOWNLOAD_TIMEOUT_SECONDS = 15
DOWNLOAD_BASE_BACKOFF_SECONDS = 30
DOWNLOAD_MAX_BACKOFF_SECONDS = 120
CACHE_DATA_PATH = "models/btc_hourly_cache.csv"

BINANCE_KLINE_LIMIT = 1000  # máximo por request

MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "btc-hourly-forecast")
MLFLOW_ARTIFACT_URI = os.getenv("MLFLOW_ARTIFACT_URI")

MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "btc_hourly_forecaster")
TAG_MODEL_VERSION = os.getenv("MLFLOW_MODEL_VERSION", "v1")
TAG_OWNER = os.getenv("MLFLOW_OWNER", "ml-team")
TAG_RISK_LEVEL = os.getenv("MLFLOW_RISK_LEVEL", "medium")
ALLOWED_RISK_LEVELS = {"low", "medium", "high", "critical"}
DVC_LOCK_PATH = "dvc.lock"
FAIRNESS_ARTIFACT_PATH = os.getenv(
    "MLFLOW_FAIRNESS_ARTIFACT_PATH",
    "evaluation/fairness_report.json",
)

REQUIRED_MLFLOW_METADATA_SCHEMA: dict[str, type[Any]] = {
    "model_name": str,
    "model_version": str,
    "model_type": str,
    "training_data_version": str,
    "metrics": dict,
    "owner": str,
    "risk_level": str,
    "fairness_checked": bool,
    "git_sha": str,
}

REQUIRED_RAW_COLUMNS = ["Close", "High", "Low", "Volume"]
REQUIRED_FEATURE_COLUMNS = [
    *FEATURE_COLUMNS,
]

RAW_DATA_SCHEMA = pa.DataFrameSchema(  # type: ignore[no-untyped-call]
    {
        "Close": pa.Column(float, nullable=False, checks=[pa.Check.gt(0)]),
        "High": pa.Column(float, nullable=False, checks=[pa.Check.gt(0)]),
        "Low": pa.Column(float, nullable=False, checks=[pa.Check.gt(0)]),
        "Volume": pa.Column(float, nullable=False, checks=[pa.Check.ge(0)]),
    },
    checks=[
        pa.Check(lambda df: (df["High"] >= df["Low"]).all(), error="High deve ser >= Low."),
        pa.Check(
            lambda df: ((df["Close"] >= df["Low"]) & (df["Close"] <= df["High"]))
            .all(),
            error="Close deve estar entre Low e High.",
        ),
    ],
    strict=True,
    coerce=True,
)

FEATURE_DATA_SCHEMA = pa.DataFrameSchema(  # type: ignore[no-untyped-call]
    {
        "log_return": pa.Column(float, nullable=False, checks=[pa.Check.in_range(-1.0, 1.0)]),
        "rsi": pa.Column(float, nullable=False, checks=[pa.Check.in_range(0.0, 1.0)]),
        "macd_signal": pa.Column(float, nullable=False, checks=[pa.Check.in_range(-2.0, 2.0)]),
        "bb_pct_b": pa.Column(float, nullable=False, checks=[pa.Check.in_range(0.0, 1.0)]),
        "sma_ratio": pa.Column(float, nullable=False, checks=[pa.Check.in_range(-1.0, 1.0)]),
        "vol_ratio": pa.Column(float, nullable=False, checks=[pa.Check.in_range(0.0, 10.0)]),
    },
    strict=True,
    coerce=True,
)


def validate_temporal_consistency(index: pd.Index, expected_frequency: str = "1h") -> None:
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("Índice temporal inválido: esperado DatetimeIndex.")

    if index.tz is None:
        raise ValueError("Índice temporal inválido: timezone obrigatório (UTC recomendado).")

    if not index.is_monotonic_increasing:
        raise ValueError("Consistência temporal inválida: índice deve estar ordenado.")

    if index.has_duplicates:
        raise ValueError("Consistência temporal inválida: timestamps duplicados detectados.")

    deltas = index.to_series().diff().dropna()
    if deltas.empty:
        return

    expected_delta = pd.Timedelta(expected_frequency)
    inconsistent_steps = deltas[deltas != expected_delta]
    if not inconsistent_steps.empty:
        raise ValueError(
            "Consistência temporal inválida: frequência horária irregular detectada. "
            f"Esperado: {expected_delta}, encontrados {inconsistent_steps.nunique()} intervalos distintos."
        )


def validate_raw_training_data(data: pd.DataFrame) -> pd.DataFrame:
    try:
        validated_data = RAW_DATA_SCHEMA.validate(data, lazy=True)
    except pa.errors.SchemaErrors as error:
        raise ValueError(f"Validação de dados brutos falhou: {error}") from error

    validate_temporal_consistency(validated_data.index)
    return validated_data


def validate_feature_training_data(features: pd.DataFrame) -> pd.DataFrame:
    try:
        validated_features = FEATURE_DATA_SCHEMA.validate(features, lazy=True)
    except pa.errors.SchemaErrors as error:
        raise ValueError(f"Validação de features falhou: {error}") from error

    validate_temporal_consistency(validated_features.index)
    return validated_features


def validate_mlflow_metadata_tags(tags: dict[str, Any], context: str) -> dict[str, Any]:
    """Valida schema obrigatório de metadata para runs e modelos no MLflow."""
    validated = dict(tags)

    missing_fields = [field for field in REQUIRED_MLFLOW_METADATA_SCHEMA if field not in validated]
    if missing_fields:
        raise ValueError(
            f"Schema de metadata MLflow inválido ({context}): campos ausentes: {missing_fields}"
        )

    invalid_fields: list[str] = []
    for field, expected_type in REQUIRED_MLFLOW_METADATA_SCHEMA.items():
        value = validated.get(field)
        if value is None:
            invalid_fields.append(f"{field}=None")
            continue

        if expected_type is str and str(value).strip() == "":
            invalid_fields.append(f"{field}=<empty>")
            continue

        if not isinstance(value, expected_type):
            invalid_fields.append(
                f"{field} (esperado {expected_type.__name__}, recebido {type(value).__name__})"
            )

    if invalid_fields:
        raise ValueError(
            f"Schema de metadata MLflow inválido ({context}): campos inválidos: {invalid_fields}"
        )

    risk_level = str(validated["risk_level"]).strip().lower()
    if risk_level not in ALLOWED_RISK_LEVELS:
        raise ValueError(
            "Schema de metadata MLflow inválido "
            f"({context}): risk_level fora do domínio permitido {sorted(ALLOWED_RISK_LEVELS)}"
        )

    metrics_payload = validated["metrics"]
    if not isinstance(metrics_payload, dict):
        raise ValueError(
            f"Schema de metadata MLflow inválido ({context}): metrics deve ser dict."
        )

    for metric_name, metric_value in metrics_payload.items():
        if not isinstance(metric_name, str) or metric_name.strip() == "":
            raise ValueError(
                f"Schema de metadata MLflow inválido ({context}): nome de métrica inválido."
            )
        if not isinstance(metric_value, (int, float, bool, str)):
            raise ValueError(
                "Schema de metadata MLflow inválido "
                f"({context}): valor de métrica inválido para '{metric_name}'."
            )

    return validated


def validate_required_training_metadata(
    *,
    model_name: str | None,
    model_version: str | None,
    training_data_version: str | None,
    model_type: str | None,
    owner: str | None,
    risk_level: str | None,
) -> dict[str, str]:
    """Valida metadados obrigatórios da função de treino antes da run MLflow.

    Args:
        model_name: Nome do modelo no Registry.
        model_version: Versão semântica definida para o treino.
        training_data_version: Versão/lineage dos dados usados no treino.
        model_type: Tipo do modelo (ex.: ``time_series``).
        owner: Responsável técnico pelo modelo.
        risk_level: Nível de risco de governança (``low|medium|high|critical``).

    Returns:
        Dicionário com as tags obrigatórias já validadas.

    Raises:
        ValueError: Se qualquer tag obrigatória estiver ausente ou vazia.
    """
    required_metadata_tags: dict[str, str | None] = {
        "model_name": model_name,
        "model_version": model_version,
        "training_data_version": training_data_version,
        "model_type": model_type,
        "owner": owner,
        "risk_level": risk_level,
    }

    missing_tags = [
        key
        for key, value in required_metadata_tags.items()
        if value is None or str(value).strip() == ""
    ]
    if missing_tags:
        raise ValueError(
            "Metadados obrigatórios ausentes para a função de treino: "
            f"{', '.join(missing_tags)}"
        )

    return {key: str(value).strip() for key, value in required_metadata_tags.items() if value is not None}


def get_git_sha() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except Exception as error:
        logger.warning("Nao foi possivel obter git SHA dinamicamente: %s", error)
        return "unknown"


def get_git_sha_required() -> str:
    git_sha = get_git_sha()
    if git_sha == "unknown":
        raise RuntimeError(
            "Falha ao capturar git SHA para lineage imutavel de dados/modelo."
        )
    return git_sha


def _normalize_repo_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def _get_dvc_output_hash(dataset_path: str, dvc_lock_path: str = DVC_LOCK_PATH) -> str:
    if not os.path.exists(dvc_lock_path):
        raise RuntimeError(
            f"Arquivo '{dvc_lock_path}' ausente. Nao foi possivel capturar hash DVC do dataset."
        )

    with open(dvc_lock_path, encoding="utf-8") as dvc_lock_file:
        lock_data = yaml.safe_load(dvc_lock_file) or {}

    stages = lock_data.get("stages")
    if not isinstance(stages, dict):
        raise RuntimeError(
            "Estrutura invalida em dvc.lock: campo 'stages' ausente ou invalido."
        )

    normalized_dataset_path = _normalize_repo_path(dataset_path)
    for stage_data in stages.values():
        if not isinstance(stage_data, dict):
            continue
        outs = stage_data.get("outs", [])
        if not isinstance(outs, list):
            continue

        for out in outs:
            if not isinstance(out, dict):
                continue

            out_path = out.get("path")
            if not isinstance(out_path, str):
                continue
            if _normalize_repo_path(out_path) != normalized_dataset_path:
                continue

            hash_name = out.get("hash")
            if not isinstance(hash_name, str):
                raise RuntimeError(
                    f"Output '{out_path}' em dvc.lock sem campo 'hash' valido."
                )

            hash_value = out.get(hash_name)
            if not isinstance(hash_value, str) or not hash_value.strip():
                raise RuntimeError(
                    f"Output '{out_path}' em dvc.lock sem valor de hash '{hash_name}'."
                )

            return hash_value.strip()

    raise RuntimeError(
        f"Dataset '{dataset_path}' nao encontrado em dvc.lock para capturar hash DVC."
    )


def build_training_data_lineage(
    dataset_path: str = CACHE_DATA_PATH,
    dvc_lock_path: str = DVC_LOCK_PATH,
) -> dict[str, str]:
    git_sha = get_git_sha_required()
    dvc_data_hash = _get_dvc_output_hash(dataset_path=dataset_path, dvc_lock_path=dvc_lock_path)
    # Em DVC, a revisao de dados e normalmente o commit Git que referencia o lockfile.
    dvc_data_rev = git_sha
    return {
        "git_sha": git_sha,
        "dvc_data_rev": dvc_data_rev,
        "dvc_data_hash": dvc_data_hash,
        "training_data_version": f"{dvc_data_rev}:{dvc_data_hash}",
    }


def get_fairness_artifact_status(
    fairness_artifact_path: str = FAIRNESS_ARTIFACT_PATH,
) -> dict[str, Any]:
    if not os.path.exists(fairness_artifact_path):
        return {
            "fairness_checked": False,
            "artifact_path": fairness_artifact_path,
            "status": "missing",
            "alert": f"missing_fairness_artifact:{fairness_artifact_path}",
        }

    try:
        with open(fairness_artifact_path, encoding="utf-8") as fairness_file:
            json.load(fairness_file)
    except Exception as error:
        return {
            "fairness_checked": False,
            "artifact_path": fairness_artifact_path,
            "status": "invalid",
            "alert": f"invalid_fairness_artifact:{fairness_artifact_path}",
            "error": str(error),
        }

    return {
        "fairness_checked": True,
        "artifact_path": fairness_artifact_path,
        "status": "valid",
        "alert": "",
    }


def ensure_directories() -> None:
    if not os.path.exists("models"):
        os.makedirs("models")


def configure_mlflow() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise OSError(
            "A variável de ambiente MLFLOW_TRACKING_URI não foi definida. "
            "Configure-a para o PostgreSQL (AWS RDS) do MLflow Tracking Server."
        )

    if tracking_uri.startswith("file://"):
        raise OSError(
            "MLFLOW_TRACKING_URI não pode usar file:// para Model Registry. "
            "Use o endpoint HTTP/HTTPS do MLflow Tracking Server com backend SQL (RDS)."
        )

    mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        if MLFLOW_ARTIFACT_URI:
            mlflow.create_experiment(
                MLFLOW_EXPERIMENT_NAME,
                artifact_location=MLFLOW_ARTIFACT_URI,
            )
        else:
            mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)

    mlflow.set_experiment(experiment_name=MLFLOW_EXPERIMENT_NAME)


CHAMPION_METRIC = "mae_price"  # métrica usada na comparação champion-challenger
MIN_IMPROVEMENT = 0.005  # melhoria mínima de 0,5 % para promover challenger
AUTO_PROMOTE_VALIDATED = os.getenv("MLFLOW_AUTO_PROMOTE_VALIDATED", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
PROMOTION_APPROVAL_ENV_VAR = "MLFLOW_PROMOTION_APPROVED"
PROMOTION_ADMIN_COMMAND_ENV_VAR = "MLFLOW_ADMIN_COMMAND"
CHAMPION_ALIAS = os.getenv("MLFLOW_CHAMPION_ALIAS", "champion")
CANDIDATE_ALIAS = os.getenv("MLFLOW_CANDIDATE_ALIAS", "candidate")
INITIAL_REGISTRY_ALIAS = os.getenv("MLFLOW_INITIAL_REGISTRY_ALIAS", "Staging")


def _is_alias_not_found_error(error: Exception) -> bool:
    message = str(error).lower()
    return "alias" in message and ("not found" in message or "does not exist" in message)


def resolve_champion_version(*, client: Any) -> Any | None:
    try:
        return client.get_model_version_by_alias(MLFLOW_MODEL_NAME, CHAMPION_ALIAS)
    except Exception as exc:
        if _is_alias_not_found_error(exc):
            return None
        raise


def evaluate_champion_challenger(challenger_mae: float) -> bool:
    """Compara challenger com champion. Retorna True se challenger deve ser promovido."""
    client = mlflow.MlflowClient()
    try:
        champion_version = resolve_champion_version(client=client)
        if champion_version is None:
            logger.info(
                "Sem champion definido por alias '%s'. Challenger será promovido automaticamente.",
                CHAMPION_ALIAS,
            )
            return True

        champion_run = client.get_run(champion_version.run_id)
        champion_mae = float(champion_run.data.metrics.get(CHAMPION_METRIC, float("inf")))

        improvement = (champion_mae - challenger_mae) / champion_mae
        logger.info(
            "Champion MAE: %.4f | Challenger MAE: %.4f | Melhoria: %.2f%%",
            champion_mae,
            challenger_mae,
            improvement * 100,
        )
        return improvement >= MIN_IMPROVEMENT

    except Exception as exc:
        logger.warning("Erro na avalia\u00e7\u00e3o champion-challenger: %s", exc)
        return False


def promote_to_production(registered_model_version: str) -> None:
    """Promove challenger via alias champion e marca o champion anterior como arquivado."""
    client = mlflow.MlflowClient()
    previous_champion = resolve_champion_version(client=client)
    if previous_champion is not None and str(previous_champion.version) != str(registered_model_version):
        client.set_model_version_tag(
            name=MLFLOW_MODEL_NAME,
            version=previous_champion.version,
            key="lifecycle_state",
            value="archived",
        )
        client.set_model_version_tag(
            name=MLFLOW_MODEL_NAME,
            version=previous_champion.version,
            key="champion_replaced_by",
            value=str(registered_model_version),
        )
        logger.info("Champion anterior marcado como arquivado (versao %s).", previous_champion.version)

    client.set_registered_model_alias(
        name=MLFLOW_MODEL_NAME,
        alias=CHAMPION_ALIAS,
        version=registered_model_version,
    )
    client.set_model_version_tag(
        name=MLFLOW_MODEL_NAME,
        version=registered_model_version,
        key="lifecycle_state",
        value="champion",
    )
    logger.info(
        "Challenger promovido por alias '%s' (versao %s).",
        CHAMPION_ALIAS,
        registered_model_version,
    )


def is_manual_promotion_approved(challenger_version: str) -> bool:
    approval_raw = os.getenv(PROMOTION_APPROVAL_ENV_VAR, "").strip().lower()
    command_raw = os.getenv(PROMOTION_ADMIN_COMMAND_ENV_VAR, "").strip().lower()

    approved_by_env = approval_raw in {"1", "true", "yes", "approved"}

    approved_by_command = False
    if command_raw:
        if ":" in command_raw:
            action, command_version = command_raw.split(":", 1)
            approved_by_command = action in {"promote", "approve"} and command_version.strip() in {
                str(challenger_version),
                "*",
            }
        else:
            approved_by_command = command_raw in {"promote", "approve"}

    if approved_by_env:
        logger.info(
            "Gate de promocao aprovado via variavel de ambiente %s.",
            PROMOTION_APPROVAL_ENV_VAR,
        )

    if approved_by_command:
        logger.info(
            "Gate de promocao aprovado via comando administrativo em %s='%s'.",
            PROMOTION_ADMIN_COMMAND_ENV_VAR,
            command_raw,
        )

    if not approved_by_env and not approved_by_command:
        logger.info(
            "Promocao bloqueada: defina %s=true ou %s=promote[:versao] para aprovacao explicita.",
            PROMOTION_APPROVAL_ENV_VAR,
            PROMOTION_ADMIN_COMMAND_ENV_VAR,
        )

    return approved_by_env or approved_by_command


def mark_challenger_as_candidate(registered_model_version: str, reason: str) -> None:
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(
        name=MLFLOW_MODEL_NAME,
        alias=CANDIDATE_ALIAS,
        version=registered_model_version,
    )
    client.set_model_version_tag(
        name=MLFLOW_MODEL_NAME,
        version=registered_model_version,
        key="lifecycle_state",
        value="candidate",
    )
    client.set_model_version_tag(
        name=MLFLOW_MODEL_NAME,
        version=registered_model_version,
        key="promotion_gate",
        value="manual_approval_required",
    )
    client.set_model_version_tag(
        name=MLFLOW_MODEL_NAME,
        version=registered_model_version,
        key="candidate_reason",
        value=reason,
    )
    logger.info(
        "Challenger salvo como candidato no alias '%s' (versao %s). Motivo: %s",
        CANDIDATE_ALIAS,
        registered_model_version,
        reason,
    )


def set_required_tags_on_active_run(tags: dict[str, Any]) -> dict[str, Any]:
    """Valida e persiste tags obrigatórias na run ativa do MLflow.

    Raises:
        ValueError: Se qualquer tag obrigatória estiver ausente ou inválida.
        RuntimeError: Se não houver run ativa no contexto.
    """
    active_run = mlflow.active_run()
    if active_run is None:
        raise RuntimeError("Nenhuma run ativa encontrada para persistir tags obrigatórias.")

    validated_tags = validate_mlflow_metadata_tags(tags, context="run")
    mlflow.set_tags(validated_tags)
    return validated_tags


def register_challenger_initial_state(registered_model_version: str) -> str:
    """Registra o modelo recém-treinado apenas como Challenger/Staging.

    Este método NÃO promove para produção; apenas define o estado inicial no Registry.
    """
    alias = INITIAL_REGISTRY_ALIAS.strip() or "Staging"
    if alias.lower() not in {"challenger", "staging"}:
        raise ValueError(
            "MLFLOW_INITIAL_REGISTRY_ALIAS inválido. Use apenas 'Challenger' ou 'Staging'."
        )

    client = mlflow.MlflowClient()
    client.set_registered_model_alias(
        name=MLFLOW_MODEL_NAME,
        alias=alias,
        version=registered_model_version,
    )
    client.set_model_version_tag(
        name=MLFLOW_MODEL_NAME,
        version=registered_model_version,
        key="lifecycle_state",
        value="challenger",
    )
    client.set_model_version_tag(
        name=MLFLOW_MODEL_NAME,
        version=registered_model_version,
        key="deployment_stage",
        value=alias.lower(),
    )
    client.set_model_version_tag(
        name=MLFLOW_MODEL_NAME,
        version=registered_model_version,
        key="promotion_status",
        value="pending_evaluation",
    )
    logger.info(
        "Modelo registrado sem promoção automática. Alias inicial '%s' para versão %s.",
        alias,
        registered_model_version,
    )
    return alias


def handle_champion_challenger_outcome(challenger_version: str, challenger_mae: float) -> str:
    challenger_beats_champion = evaluate_champion_challenger(challenger_mae=challenger_mae)
    if challenger_beats_champion:
        if AUTO_PROMOTE_VALIDATED:
            promote_to_production(challenger_version)
            return "promoted_auto"

        if is_manual_promotion_approved(challenger_version=challenger_version):
            promote_to_production(challenger_version)
            return "promoted"

        mark_challenger_as_candidate(
            challenger_version,
            reason="metric_gate_passed_manual_approval_pending",
        )
        return "candidate_pending_approval"

    mark_challenger_as_candidate(
        challenger_version,
        reason="metric_gate_not_passed",
    )
    return "candidate_not_promoted"


def archive_challenger(registered_model_version: str) -> None:
    """Mantem challenger como candidato quando nao supera champion."""
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(
        name=MLFLOW_MODEL_NAME,
        alias=CANDIDATE_ALIAS,
        version=registered_model_version,
    )
    client.set_model_version_tag(
        name=MLFLOW_MODEL_NAME,
        version=registered_model_version,
        key="lifecycle_state",
        value="candidate",
    )
    logger.info(
        "Champion mantido. Challenger marcado como candidato no alias '%s' (versao %s).",
        CANDIDATE_ALIAS,
        registered_model_version,
    )


def log_training_artifacts(
    model: Any,
    scaler_all: Any,
    scaler_return: Any,
    metadata: dict[str, Any],
    metadata_tags: dict[str, Any],
) -> str:
    with tempfile.TemporaryDirectory(prefix="mlflow_artifacts_") as temp_dir:
        scaler_file = os.path.join(temp_dir, "scaler_btc.gz")
        scaler_return_file = os.path.join(temp_dir, "scaler_btc_return.gz")
        metadata_file = os.path.join(temp_dir, "model_metadata_btc.json")

        joblib.dump(scaler_all, scaler_file)
        joblib.dump(scaler_return, scaler_return_file)

        with open(metadata_file, "w", encoding="utf-8") as meta_file:
            json.dump(metadata, meta_file, indent=2, ensure_ascii=False)

        mlflow.keras.log_model(model, artifact_path="model")
        mlflow.log_artifact(scaler_file, artifact_path="scalers")
        mlflow.log_artifact(scaler_return_file, artifact_path="scalers")
        mlflow.log_artifact(metadata_file, artifact_path="metadata")

        active_run = mlflow.active_run()
        if active_run is None:
            raise RuntimeError("Nenhuma run ativa encontrada para registrar o modelo no Registry.")

        model_uri = f"runs:/{active_run.info.run_id}/model"
        validated_registry_tags = validate_mlflow_metadata_tags(
            metadata_tags,
            context="model_registry",
        )
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=MLFLOW_MODEL_NAME,
            tags={**validated_registry_tags, "stage": "challenger"},
        )
        logger.info(
            "Modelo registrado no Registry: %s vers\u00e3o %s",
            registered_model.name,
            registered_model.version,
        )
        return registered_model.version


def normalize_download_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs(TICKER, axis=1, level=1)
        except KeyError:
            df.columns = df.columns.get_level_values(0)

    missing_columns = [column for column in REQUIRED_RAW_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Colunas ausentes na resposta da API: {missing_columns}")

    normalized = df[REQUIRED_RAW_COLUMNS].copy().dropna()
    normalized = normalized[~normalized.index.duplicated(keep="last")]
    normalized = normalized.sort_index()
    return normalized


def load_cached_data() -> pd.DataFrame:
    if not os.path.exists(CACHE_DATA_PATH):
        return pd.DataFrame()

    try:
        cached_df = pd.read_csv(CACHE_DATA_PATH, index_col=0, parse_dates=True)
        cached_df.index.name = None
        normalized = normalize_download_dataframe(cached_df)
        if normalized.empty:
            return pd.DataFrame()
        return normalized
    except Exception as error:
        logger.warning("Falha ao ler cache local em '%s': %s", CACHE_DATA_PATH, error)
        return pd.DataFrame()


def save_cached_data(data: pd.DataFrame) -> None:
    try:
        data.to_csv(CACHE_DATA_PATH)
    except Exception as error:
        logger.warning("Nao foi possivel salvar cache local em '%s': %s", CACHE_DATA_PATH, error)


def download_from_binance() -> pd.DataFrame:
    """Baixa dados horários do BTC via Binance REST API pública.

    Usa paginação para cobrir todo o período definido em PERIOD.
    """
    logger.info("Tentando fonte alternativa: Binance REST API...")
    interval_ms = 60 * 60 * 1000  # 1 hora em ms
    days = int(PERIOD.replace("d", ""))
    end_ts = int(time.time() * 1000)
    start_ts = end_ts - days * 24 * interval_ms

    all_rows = []
    cursor = start_ts
    while cursor < end_ts:
        resp = requests.get(
            BINANCE_API_URL,
            params={
                "symbol": BINANCE_SYMBOL,
                "interval": "1h",
                "startTime": cursor,
                "endTime": end_ts,
                "limit": BINANCE_KLINE_LIMIT,
            },
            timeout=BINANCE_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        all_rows.extend(batch)
        cursor = batch[-1][0] + interval_ms  # avança para após o último candle
        time.sleep(0.2)  # respeita rate limit da Binance

    if not all_rows:
        raise ValueError("Binance retornou dados vazios.")

    timestamps = pd.to_datetime([row[0] for row in all_rows], unit="ms", utc=True)
    df = pd.DataFrame(
        {
            "Close": pd.to_numeric([row[4] for row in all_rows]),
            "High": pd.to_numeric([row[2] for row in all_rows]),
            "Low": pd.to_numeric([row[3] for row in all_rows]),
            "Volume": pd.to_numeric([row[5] for row in all_rows]),
        },
        index=timestamps,
    )
    df.index.name = "Datetime"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    logger.info("Binance: %s candles obtidos via paginacao.", len(df))
    return df


def download_crypto_data() -> pd.DataFrame:
    """Baixa dados horários do BTC no Yahoo Finance."""
    logger.info("Baixando dados horarios (%s) para %s (Ultimos %s)...", INTERVAL, TICKER, PERIOD)

    data = pd.DataFrame()
    last_error = None
    for attempt in range(1, DOWNLOAD_MAX_RETRIES + 1):
        try:
            download_df = yf.download(
                TICKER,
                period=PERIOD,
                interval=INTERVAL,
                progress=False,
                auto_adjust=False,
                threads=False,
                timeout=DOWNLOAD_TIMEOUT_SECONDS,
            )
            data = normalize_download_dataframe(download_df)
            if not data.empty:
                break
        except Exception as error:
            last_error = error
            logger.warning("Tentativa %s/%s falhou: %s", attempt, DOWNLOAD_MAX_RETRIES, error)

        if not data.empty:
            break

        if attempt < DOWNLOAD_MAX_RETRIES:
            backoff_seconds = min(
                DOWNLOAD_MAX_BACKOFF_SECONDS, DOWNLOAD_BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
            )
            logger.info("Aguardando %ss antes da proxima tentativa...", backoff_seconds)
            time.sleep(backoff_seconds)

    if data.empty:
        cached_data = load_cached_data()
        if not cached_data.empty:
            logger.warning(
                "API indisponivel/limitada. Usando cache local em '%s' com %s registros.",
                CACHE_DATA_PATH,
                len(cached_data),
            )
            return cached_data

        # Fallback: Binance
        try:
            binance_data = download_from_binance()
            save_cached_data(binance_data)
            logger.info("Total de registros (Binance): %s", len(binance_data))
            return binance_data
        except Exception as binance_err:
            logger.warning("Binance tambem falhou: %s", binance_err)

        raise ValueError(
            "A API retornou um DataFrame vazio após "
            f"{DOWNLOAD_MAX_RETRIES} tentativas e não há cache local disponível. "
            f"Último erro: {last_error}"
        )

    save_cached_data(data)

    logger.info("Total de registros (horas): %s", len(data))
    return data


def build_feature_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """Wrapper de compatibilidade para o módulo compartilhado de features."""
    return _build_feature_matrix(data)


def create_sliding_window_multifeature(
    dataset: NDArray[np.float64], look_back: int = 60
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Cria janelas deslizantes para entrada multi-feature.
    dataset: shape (N, n_features)
    Retorna X de shape (samples, look_back, n_features), y de shape (samples,)
    onde y é o log_return da posição [look_back] (feature índice 0).
    """
    X, y = [], []
    for i in range(look_back, len(dataset)):
        X.append(dataset[i - look_back : i, :])  # janela completa com todas as features
        y.append(dataset[i, 0])  # target: log_return (índice 0)
    return np.array(X), np.array(y)


def safe_mape(y_true: NDArray[np.float64], y_pred: NDArray[np.float64], eps: float = 1e-8) -> float:
    denominator = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100)


def build_lstm_architecture(input_shape: tuple[int, int]) -> Sequential:
    """Modelo LSTM bidirecional com múltiplas features para melhor acurácia direcional."""
    model = Sequential(
        [
            Input(shape=input_shape),
            Bidirectional(LSTM(units=64, return_sequences=True)),
            Dropout(0.2),
            LSTM(units=48, return_sequences=True),
            Dropout(0.2),
            LSTM(units=32, return_sequences=False),
            Dropout(0.2),
            Dense(units=16, activation="relu"),
            Dense(units=1),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mean_squared_error")
    return model


def run_walk_forward_backtest(
    X_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    scaler_return: Any,
) -> None:
    """Walk-forward backtest com modelo multi-feature."""
    if len(X_train) < (WALK_FORWARD_SPLITS + 1):
        logger.warning("Dados insuficientes para walk-forward. Backtest pulado.")
        return

    logger.info("Iniciando walk-forward backtest com %s splits...", WALK_FORWARD_SPLITS)
    tscv = TimeSeriesSplit(n_splits=WALK_FORWARD_SPLITS)
    model_maes = []
    baseline_maes = []

    for fold_idx, (tr_idx, val_idx) in enumerate(tscv.split(X_train), start=1):
        X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
        X_val_fold, y_val_fold = X_train[val_idx], y_train[val_idx]

        fold_model = build_lstm_architecture((X_train.shape[1], X_train.shape[2]))
        fold_early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

        fold_model.fit(
            X_tr,
            y_tr,
            batch_size=BATCH_SIZE,
            epochs=WALK_FORWARD_EPOCHS,
            validation_data=(X_val_fold, y_val_fold),
            callbacks=[fold_early_stop],
            verbose=0,
        )

        y_pred_scaled = fold_model.predict(X_val_fold, verbose=0).reshape(-1, 1)
        y_pred = scaler_return.inverse_transform(y_pred_scaled).reshape(-1)

        y_real_scaled = y_val_fold.reshape(-1, 1)
        y_real = scaler_return.inverse_transform(y_real_scaled).reshape(-1)

        # baseline: último log_return da janela (índice 0 da última step)
        baseline_scaled = X_val_fold[:, -1, 0].reshape(-1, 1)
        baseline_pred = scaler_return.inverse_transform(baseline_scaled).reshape(-1)

        fold_mae = mean_absolute_error(y_real, y_pred)
        fold_baseline_mae = mean_absolute_error(y_real, baseline_pred)

        model_maes.append(fold_mae)
        baseline_maes.append(fold_baseline_mae)

        logger.info(
            "[WF][Fold %s] MAE modelo (retorno): %.6f | MAE baseline (retorno): %.6f",
            fold_idx,
            fold_mae,
            fold_baseline_mae,
        )

    logger.info(
        "[WF][Media] MAE modelo (retorno): %.6f | MAE baseline (retorno): %.6f",
        np.mean(model_maes),
        np.mean(baseline_maes),
    )


def main() -> None:
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    ensure_directories()
    configure_mlflow()

    data_lineage = build_training_data_lineage(dataset_path=CACHE_DATA_PATH)
    fairness_status = get_fairness_artifact_status()

    required_metadata_tags = validate_required_training_metadata(
        model_name=MLFLOW_MODEL_NAME,
        model_version=TAG_MODEL_VERSION,
        training_data_version=data_lineage.get("training_data_version"),
        model_type="time_series",
        owner=TAG_OWNER,
        risk_level=TAG_RISK_LEVEL,
    )

    run_tags = {
        "model_name": required_metadata_tags["model_name"],
        "model_version": required_metadata_tags["model_version"],
        "model_type": required_metadata_tags["model_type"],
        "metrics": {},
        "owner": required_metadata_tags["owner"],
        "risk_level": required_metadata_tags["risk_level"],
        "training_data_version": required_metadata_tags["training_data_version"],
        "git_sha": data_lineage["git_sha"],
        "dvc_data_rev": data_lineage["dvc_data_rev"],
        "dvc_data_hash": data_lineage["dvc_data_hash"],
        "fairness_checked": fairness_status["fairness_checked"],
    }

    params = {
        "ticker": TICKER,
        "period": PERIOD,
        "interval": INTERVAL,
        "lookback": LOOKBACK,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "test_size_pct": TEST_SIZE_PCT,
        "val_size_pct": VAL_SIZE_PCT,
        "walk_forward_splits": WALK_FORWARD_SPLITS,
        "walk_forward_epochs": WALK_FORWARD_EPOCHS,
        "random_seed": RANDOM_SEED,
        "epsilon": EPSILON,
        "download_max_retries": DOWNLOAD_MAX_RETRIES,
        "download_timeout_seconds": DOWNLOAD_TIMEOUT_SECONDS,
        "download_base_backoff_seconds": DOWNLOAD_BASE_BACKOFF_SECONDS,
        "download_max_backoff_seconds": DOWNLOAD_MAX_BACKOFF_SECONDS,
        "architecture": "bidirectional_lstm_multifeature",
        "optimizer": "Adam",
        "learning_rate": 1e-3,
        "dvc_data_rev": data_lineage["dvc_data_rev"],
        "dvc_data_hash": data_lineage["dvc_data_hash"],
        "training_data_version": data_lineage["training_data_version"],
        "git_sha": data_lineage["git_sha"],
    }

    with mlflow.start_run(run_name=f"{TICKER}_{INTERVAL}_training"):
        validated_run_tags = set_required_tags_on_active_run(run_tags)
        mlflow.log_params(params)
        mlflow.set_tag("fairness_artifact_path", fairness_status["artifact_path"])
        mlflow.set_tag("fairness_artifact_status", fairness_status["status"])

        if fairness_status["fairness_checked"]:
            mlflow.log_metric("fairness_artifact_present", 1.0)
            mlflow.log_artifact(fairness_status["artifact_path"], artifact_path="fairness")
        else:
            mlflow.log_metric("fairness_artifact_present", 0.0)
            mlflow.set_tag("fairness_alert", fairness_status["alert"])
            if fairness_status.get("error"):
                mlflow.set_tag("fairness_alert_error", str(fairness_status["error"]))
            logger.warning(
                "Fairness artefato ausente/invalido. fairness_checked=False. Detalhes: %s",
                fairness_status,
            )

        # 1. Download e feature engineering
        raw_data = validate_raw_training_data(download_crypto_data())
        features_df = validate_feature_training_data(build_feature_matrix(raw_data))
        n_features = features_df.shape[1]
        close_series = raw_data["Close"].reindex(features_df.index)
        mlflow.log_param("n_features", int(n_features))
        mlflow.log_param("features", ",".join(list(features_df.columns)))
        mlflow.log_metric("total_rows", float(len(raw_data)))
        mlflow.log_metric("feature_rows", float(len(features_df)))

        logger.info("Features utilizadas (%s): %s", n_features, list(features_df.columns))

        # 2. Split temporal treino/teste
        split_idx = int(len(features_df) * (1 - TEST_SIZE_PCT))
        train_features = features_df.iloc[:split_idx]
        test_features = features_df.iloc[split_idx:]

        if len(train_features) <= LOOKBACK:
            raise ValueError(
                "Dados de treino insuficientes. "
                f"Necessário mais que {LOOKBACK} registros, "
                f"recebido: {len(train_features)}."
            )
        if len(test_features) == 0:
            raise ValueError("Conjunto de teste vazio. Ajuste TEST_SIZE_PCT.")

        logger.info("Treino: %s horas | Teste: %s horas", len(train_features), len(test_features))
        mlflow.log_metric("train_rows", float(len(train_features)))
        mlflow.log_metric("test_rows", float(len(test_features)))

        # 3. Scaling por feature (scaler_all) + scaler separado para log_return (para inversão)
        scaler_all = MinMaxScaler(feature_range=(0, 1))
        scaled_train = scaler_all.fit_transform(train_features.values)

        # Scaler exclusivo para log_return (feature 0) — usado na inferência e métricas
        scaler_return = MinMaxScaler(feature_range=(0, 1))
        scaler_return.fit(train_features[["log_return"]].values)

        # Escalar conjunto total para criar janelas de teste
        all_features = pd.concat([train_features, test_features], axis=0)
        scaled_all = scaler_all.transform(all_features.values)

        # 4. Criar janelas deslizantes
        X_train, y_train = create_sliding_window_multifeature(scaled_train, LOOKBACK)
        X_all, y_all = create_sliding_window_multifeature(scaled_all, LOOKBACK)

        test_start_idx_in_windows = split_idx - LOOKBACK
        if test_start_idx_in_windows < 0:
            raise ValueError(
                "Split inválido para LOOKBACK atual. Ajuste TEST_SIZE_PCT ou LOOKBACK."
            )

        X_test = X_all[test_start_idx_in_windows:]
        y_test = y_all[test_start_idx_in_windows:]

        # 5. Walk-forward backtest
        run_walk_forward_backtest(X_train, y_train, scaler_return)

        # 6. Treino final com validação
        val_size = max(1, int(len(X_train) * VAL_SIZE_PCT))
        if val_size >= len(X_train):
            val_size = 1

        X_train_fit = X_train[:-val_size]
        y_train_fit = y_train[:-val_size]
        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]

        if len(X_train_fit) == 0:
            raise ValueError("Treino ficou vazio após split de validação. Ajuste VAL_SIZE_PCT.")

        model = build_lstm_architecture((LOOKBACK, n_features))

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6, verbose=0),
        ]

        model.fit(
            X_train_fit,
            y_train_fit,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=0,
        )

        # 7. Avaliação no conjunto de teste
        predictions_scaled = model.predict(X_test, verbose=0).reshape(-1, 1)
        predictions_return = scaler_return.inverse_transform(predictions_scaled).reshape(-1)
        y_test_return = scaler_return.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)

        target_indices = all_features.index[
            LOOKBACK + test_start_idx_in_windows : LOOKBACK
            + test_start_idx_in_windows
            + len(y_test)
        ]
        prev_close = close_series.shift(1).reindex(target_indices).values
        y_test_real_price = close_series.reindex(target_indices).values

        valid_mask = (~np.isnan(prev_close)) & (~np.isnan(y_test_real_price))
        prev_close = prev_close[valid_mask]
        y_test_real_price = y_test_real_price[valid_mask]
        predictions_return = predictions_return[valid_mask]
        y_test_return = y_test_return[valid_mask]

        predictions_price = prev_close * np.exp(predictions_return)
        baseline_predictions_price = prev_close

        baseline_scaled = X_test[:, -1, 0].reshape(-1, 1)
        baseline_return = scaler_return.inverse_transform(baseline_scaled).reshape(-1)[valid_mask]

        mae = mean_absolute_error(y_test_real_price, predictions_price)
        rmse = np.sqrt(mean_squared_error(y_test_real_price, predictions_price))
        mape = safe_mape(y_test_real_price, predictions_price, EPSILON)

        baseline_mae = mean_absolute_error(y_test_real_price, baseline_predictions_price)
        baseline_rmse = np.sqrt(mean_squared_error(y_test_real_price, baseline_predictions_price))
        baseline_mape = safe_mape(y_test_real_price, baseline_predictions_price, EPSILON)

        model_return_mae = mean_absolute_error(y_test_return, predictions_return)
        baseline_return_mae = mean_absolute_error(y_test_return, baseline_return)

        model_direction = np.sign(predictions_return)
        real_direction = np.sign(y_test_return)
        direction_accuracy = np.mean(model_direction == real_direction) * 100

        beats_baseline = mae < baseline_mae and rmse < baseline_rmse

        mlflow.log_metrics(
            {
                "mae_price": float(mae),
                "rmse_price": float(rmse),
                "mape_price": float(mape),
                "mae_price_baseline": float(baseline_mae),
                "rmse_price_baseline": float(baseline_rmse),
                "mape_price_baseline": float(baseline_mape),
                "mae_return": float(model_return_mae),
                "mae_return_baseline": float(baseline_return_mae),
                "direction_accuracy_pct": float(direction_accuracy),
                "beats_baseline": float(beats_baseline),
            }
        )

        governance_metrics = {
            "mae_price": float(mae),
            "rmse_price": float(rmse),
            "mape_price": float(mape),
            "mae_price_baseline": float(baseline_mae),
            "rmse_price_baseline": float(baseline_rmse),
            "mape_price_baseline": float(baseline_mape),
            "mae_return": float(model_return_mae),
            "mae_return_baseline": float(baseline_return_mae),
            "direction_accuracy_pct": float(direction_accuracy),
            "beats_baseline": bool(beats_baseline),
        }

        governance_tags = {
            **validated_run_tags,
            "metrics": governance_metrics,
        }
        validated_governance_tags = validate_mlflow_metadata_tags(
            governance_tags,
            context="run_post_training",
        )
        mlflow.set_tags(validated_governance_tags)
        logger.info(
            "governance_tags_validated | event=%s payload=%s",
            "mlflow_governance_schema_enforced",
            json.dumps(validated_governance_tags, ensure_ascii=True),
        )

        # 8. Registrar artefatos e metadados no MLflow
        metadata = {
            "ticker": TICKER,
            "target": "log_return",
            "lookback": LOOKBACK,
            "interval": INTERVAL,
            "period": PERIOD,
            "seed": RANDOM_SEED,
            "n_features": n_features,
            "features": list(features_df.columns),
            "architecture": "bidirectional_lstm_multifeature",
            "metrics": {
                "mae_price": float(mae),
                "rmse_price": float(rmse),
                "mape_price": float(mape),
                "mae_price_baseline": float(baseline_mae),
                "rmse_price_baseline": float(baseline_rmse),
                "mape_price_baseline": float(baseline_mape),
                "mae_return": float(model_return_mae),
                "mae_return_baseline": float(baseline_return_mae),
                "direction_accuracy_pct": float(direction_accuracy),
            },
            "beats_baseline": bool(beats_baseline),
        }

        challenger_version = log_training_artifacts(
            model,
            scaler_all,
            scaler_return,
            metadata,
            metadata_tags=validated_governance_tags,
        )
        logger.info("Artefatos registrados no MLflow (model/.keras, scalers/.gz e metadata).")

        # 9. Sem promoção automática: mantém somente estado inicial Challenger/Staging.
        initial_alias = register_challenger_initial_state(challenger_version)
        logger.info("Estado inicial aplicado no Registry: %s", initial_alias)

        logger.info("\n%s", "=" * 40)
        logger.info("RELATORIO DE PERFORMANCE (%s - HORARIO)", TICKER)
        logger.info("%s", "=" * 40)
        logger.info("Features: %s", list(features_df.columns))
        logger.info("Erro Medio Absoluto (MAE): $ %.2f", mae)
        logger.info("RMSE: $ %.2f", rmse)
        logger.info("MAPE: %.2f%%", mape)
        logger.info("%s", "-" * 40)
        logger.info("BASELINE INGENUO (y_hat = ultimo close da janela)")
        logger.info("MAE Baseline: $ %.2f", baseline_mae)
        logger.info("RMSE Baseline: $ %.2f", baseline_rmse)
        logger.info("MAPE Baseline: %.2f%%", baseline_mape)
        logger.info("%s", "-" * 40)
        logger.info("METRICAS DE RETORNO E DIRECAO")
        logger.info("MAE Retorno (Modelo): %.6f", model_return_mae)
        logger.info("MAE Retorno (Baseline): %.6f", baseline_return_mae)
        logger.info("Acuracia Direcional: %.2f%%", direction_accuracy)
        logger.info("%s", "-" * 40)
        logger.info("Modelo superou baseline? %s", "SIM" if beats_baseline else "NAO")
        logger.info("%s", "=" * 40)
        active_run = mlflow.active_run()
        run_id = active_run.info.run_id if active_run is not None else "unknown"
        logger.info("MLflow run finalizada: %s", run_id)

        # stdout estrito para Step Functions: apenas JSON com run_id e model_version.
        output_payload = {
            "run_id": run_id,
            "model_version": str(challenger_version),
        }
        print(json.dumps(output_payload, ensure_ascii=True))


if __name__ == "__main__":
    main()
