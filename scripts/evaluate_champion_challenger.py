"""Validação Champion-Challenger rigorosa com quality gate estatístico (AUC).

Este script é executado como task ECS dedicada no pipeline Step Functions e
implementa a política de promoção exigida no Datathon (Gap 07):

1. Conecta ao MLflow Tracking URI (RDS suportado via backend SQLAlchemy).
2. Resolve o challenger recém-treinado a partir do alias de candidato.
3. Faz download do Champion em produção no Model Registry (alias Production)
   e do Challenger via run_id.
4. Executa predições em holdout isolado e calcula AUC para ambos.
5. Calcula delta_auc = challenger_auc - champion_auc.
6. Só promove para alias Production quando delta_auc >= 0.005.

Exit codes (consumidos pela Step Functions):
    0  -> Challenger promovido para Production.
    10 -> Challenger rejeitado no quality gate (não é erro de infra).
    1  -> Erro de infraestrutura/configuração/API.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger("stockcast.champion_challenger")

# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------
EXIT_PROMOTED = 0
EXIT_NOT_PROMOTED = 10
EXIT_SYSTEM_ERROR = 1

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "btc_hourly_forecaster")
CHAMPION_ALIAS = os.getenv("MLFLOW_CHAMPION_ALIAS", "champion")
CANDIDATE_ALIAS = os.getenv("MLFLOW_CANDIDATE_ALIAS", "candidate")
PRODUCTION_ALIAS = os.getenv("MLFLOW_PRODUCTION_ALIAS", "Production")

HOLDOUT_DATA_PATH = os.getenv(
    "CHAMPION_CHALLENGER_HOLDOUT_PATH",
    "data/processed/champion_challenger_holdout.csv",
)
HOLDOUT_TARGET_COLUMN = os.getenv("CHAMPION_CHALLENGER_TARGET_COLUMN", "target")
MODEL_ARTIFACT_PATH = os.getenv("MLFLOW_MODEL_ARTIFACT_PATH", "model")

_min_improvement_raw = os.getenv("CHAMPION_MIN_IMPROVEMENT", "0.005")
MIN_IMPROVEMENT: float = float(_min_improvement_raw)


def _configure_mlflow() -> Any:
    """Configura o cliente MLflow com Tracking URI suportando backend em AWS RDS.

    Returns:
        Módulo ``mlflow`` configurado e pronto para uso.

    Raises:
        RuntimeError: Se ``MLFLOW_TRACKING_URI`` estiver ausente.
    """
    import mlflow  # type: ignore[import-untyped]

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if not tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI não está definido.")
    mlflow.set_tracking_uri(tracking_uri)
    logger.info("MLflow tracking URI configurado: %s", tracking_uri)
    return mlflow


def _resolve_candidate_version(client: Any) -> tuple[str, str]:
    """Resolve versão e run_id do Challenger a partir do alias de candidato.

    Args:
        client: Instância de ``mlflow.MlflowClient``.

    Returns:
        Tupla ``(version, run_id)`` do challenger.

    Raises:
        RuntimeError: Se o alias de candidato não existir.
    """
    try:
        model_version = client.get_model_version_by_alias(MLFLOW_MODEL_NAME, CANDIDATE_ALIAS)
    except Exception as exc:
        raise RuntimeError(
            f"Alias '{CANDIDATE_ALIAS}' não encontrado para '{MLFLOW_MODEL_NAME}': {exc}"
        ) from exc

    version = str(model_version.version)
    run_id = str(model_version.run_id)
    logger.info("Challenger identificado: version=%s run_id=%s", version, run_id)
    return version, run_id


def download_champion_model_from_registry(
    *,
    mlflow_module: Any,
    model_name: str,
    production_alias: str,
) -> tuple[Any, str, str]:
    """Faz download do Champion em produção via Model Registry do MLflow.

    Args:
        mlflow_module: Módulo ``mlflow`` já configurado no tracking URI.
        model_name: Nome do modelo registrado.
        production_alias: Alias que representa produção (ex.: ``Production``).

    Returns:
        Tupla ``(champion_model, champion_version, champion_run_id)``.

    Raises:
        RuntimeError: Se não houver Champion em produção ou se o download falhar.
    """
    client = mlflow_module.MlflowClient()
    try:
        champion_version = client.get_model_version_by_alias(model_name, production_alias)
    except Exception as exc:
        raise RuntimeError(
            f"Champion de produção não encontrado no alias '{production_alias}': {exc}"
        ) from exc

    uri = f"models:/{model_name}@{production_alias}"
    try:
        champion_model = mlflow_module.pyfunc.load_model(uri)
    except Exception as exc:
        raise RuntimeError(f"Falha ao carregar Champion em '{uri}': {exc}") from exc

    return champion_model, str(champion_version.version), str(champion_version.run_id)


def load_challenger_model_by_run_id(
    *,
    mlflow_module: Any,
    challenger_run_id: str,
    model_artifact_path: str,
) -> Any:
    """Carrega o Challenger recém-treinado a partir do run_id.

    Args:
        mlflow_module: Módulo ``mlflow`` configurado.
        challenger_run_id: Run ID do challenger.
        model_artifact_path: Caminho do artefato do modelo dentro do run.

    Returns:
        Modelo carregado via ``mlflow.pyfunc``.

    Raises:
        RuntimeError: Se o modelo não puder ser carregado.
    """
    uri = f"runs:/{challenger_run_id}/{model_artifact_path}"
    try:
        return mlflow_module.pyfunc.load_model(uri)
    except Exception as exc:
        raise RuntimeError(f"Falha ao carregar Challenger em '{uri}': {exc}") from exc


def _load_holdout_dataset(path: str, target_column: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Carrega holdout isolado para validação Champion-Challenger.

    Args:
        path: Caminho CSV do holdout.
        target_column: Nome da coluna alvo binária.

    Returns:
        Tupla ``(X_holdout, y_holdout)``.

    Raises:
        RuntimeError: Se dataset/coluna não estiver disponível.
    """
    if not os.path.exists(path):
        raise RuntimeError(f"Holdout set não encontrado: {path}")

    holdout_df = pd.read_csv(path)
    if target_column not in holdout_df.columns:
        raise RuntimeError(
            f"Coluna alvo '{target_column}' ausente no holdout. Colunas: {list(holdout_df.columns)}"
        )

    y_true = holdout_df[target_column].to_numpy()
    X_holdout = holdout_df.drop(columns=[target_column])
    return X_holdout, y_true


def _predict_scores(model: Any, X_holdout: pd.DataFrame) -> np.ndarray:
    """Gera vetor de score para cálculo de AUC.

    Args:
        model: Modelo carregado via ``mlflow.pyfunc``.
        X_holdout: Features do holdout.

    Returns:
        Array 1D com scores contínuos.
    """
    predictions = model.predict(X_holdout)
    pred_array = np.asarray(predictions)

    if pred_array.ndim == 2 and pred_array.shape[1] > 1:
        # Convenção: coluna 1 representa score da classe positiva.
        return pred_array[:, 1].astype(float)

    if pred_array.ndim == 2 and pred_array.shape[1] == 1:
        return pred_array[:, 0].astype(float)

    return pred_array.reshape(-1).astype(float)


def _resolve_model_version_by_run_id(client: Any, run_id: str) -> str:
    """Resolve a versão registrada no Model Registry associada a um run_id.

    Args:
        client: Instância ``mlflow.MlflowClient``.
        run_id: Run ID do challenger.

    Returns:
        Número da versão registrada como string.

    Raises:
        RuntimeError: Se nenhuma versão associada for encontrada.
    """
    versions = client.search_model_versions(f"name='{MLFLOW_MODEL_NAME}'")
    candidates = [v for v in versions if str(v.run_id) == run_id]
    if not candidates:
        raise RuntimeError(
            f"Nenhuma model version encontrada para run_id={run_id} em {MLFLOW_MODEL_NAME}"
        )
    selected = max(candidates, key=lambda item: int(item.version))
    return str(selected.version)


def _promote_to_production_alias(client: Any, challenger_version: str, delta_auc: float) -> None:
    """Promove challenger para alias Production e registra tags de governança."""
    client.set_registered_model_alias(
        name=MLFLOW_MODEL_NAME,
        alias=PRODUCTION_ALIAS,
        version=challenger_version,
    )
    client.set_model_version_tag(
        name=MLFLOW_MODEL_NAME,
        version=challenger_version,
        key="lifecycle_state",
        value="production",
    )
    client.set_model_version_tag(
        name=MLFLOW_MODEL_NAME,
        version=challenger_version,
        key="promotion_gate",
        value="auc_delta_passed",
    )
    client.set_model_version_tag(
        name=MLFLOW_MODEL_NAME,
        version=challenger_version,
        key="delta_auc",
        value=f"{delta_auc:.6f}",
    )


def _mark_rejected(client: Any, challenger_version: str, reason: str) -> None:
    """Marca challenger como rejeitado no gate estatístico."""
    client.set_model_version_tag(
        name=MLFLOW_MODEL_NAME,
        version=challenger_version,
        key="promotion_gate",
        value="auc_delta_rejected",
    )
    client.set_model_version_tag(
        name=MLFLOW_MODEL_NAME,
        version=challenger_version,
        key="rejection_reason",
        value=reason,
    )


def main() -> int:
    """Executa validação Champion-Challenger baseada em AUC + quality gate.

    Returns:
        Exit code para integração com Step Functions.
    """
    try:
        mlflow_module = _configure_mlflow()
        client = mlflow_module.MlflowClient()

        challenger_version, challenger_run_id = _resolve_candidate_version(client)
        challenger_model = load_challenger_model_by_run_id(
            mlflow_module=mlflow_module,
            challenger_run_id=challenger_run_id,
            model_artifact_path=MODEL_ARTIFACT_PATH,
        )

        X_holdout, y_holdout = _load_holdout_dataset(
            path=HOLDOUT_DATA_PATH,
            target_column=HOLDOUT_TARGET_COLUMN,
        )

        challenger_scores = _predict_scores(challenger_model, X_holdout)
        challenger_auc = float(roc_auc_score(y_holdout, challenger_scores))

        # Tenta champion em produção; se não houver, promove challenger automaticamente.
        champion_auc: float | None = None
        champion_version: str | None = None
        champion_run_id: str | None = None
        try:
            champion_model, champion_version, champion_run_id = (
                download_champion_model_from_registry(
                    mlflow_module=mlflow_module,
                    model_name=MLFLOW_MODEL_NAME,
                    production_alias=PRODUCTION_ALIAS,
                )
            )
            champion_scores = _predict_scores(champion_model, X_holdout)
            champion_auc = float(roc_auc_score(y_holdout, champion_scores))
            delta_auc = challenger_auc - champion_auc
        except RuntimeError:
            delta_auc = float("inf")

        logger.info(
            "Evaluation summary | champion_auc=%s challenger_auc=%.6f delta_auc=%s threshold=%.6f",
            f"{champion_auc:.6f}" if champion_auc is not None else "None",
            challenger_auc,
            f"{delta_auc:.6f}" if np.isfinite(delta_auc) else "inf",
            MIN_IMPROVEMENT,
        )

        if delta_auc >= MIN_IMPROVEMENT:
            _promote_to_production_alias(client, challenger_version, float(delta_auc))
            logger.info(
                "APPROVED: challenger promoted to alias '%s' | challenger_version=%s",
                PRODUCTION_ALIAS,
                challenger_version,
            )
            return EXIT_PROMOTED

        reason = (
            f"quality_gate_failed delta_auc={delta_auc:.6f} < min_required={MIN_IMPROVEMENT:.6f}"
        )
        _mark_rejected(client, challenger_version, reason)
        logger.info(
            "REJECTED: challenger not promoted | challenger_version=%s reason=%s",
            challenger_version,
            reason,
        )
        return EXIT_NOT_PROMOTED

    except Exception as exc:
        logger.error("SYSTEM_ERROR during champion-challenger evaluation: %s", exc)
        return EXIT_SYSTEM_ERROR


if __name__ == "__main__":
    sys.exit(main())
