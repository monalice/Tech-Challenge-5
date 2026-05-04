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
from sklearn.metrics import roc_auc_score

from src.adapters.ml.s3_model_manager import S3ModelManager

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
FALLBACK_CANDIDATE_ALIASES = ["Staging", "challenger"]

_min_improvement_raw = os.getenv("CHAMPION_MIN_IMPROVEMENT", "0.005")
MIN_IMPROVEMENT: float = float(_min_improvement_raw)

# ---------------------------------------------------------------------------
# S3 model paths
# ---------------------------------------------------------------------------
S3_MODELS_BUCKET = os.getenv("S3_MODELS_BUCKET", "").strip()
S3_CHAMPION_PREFIX = "champion"
S3_CHALLENGER_PREFIX = "challenger"
S3_MODEL_FILENAME = "lstm_btc_hourly.keras"
S3_SCALER_FILENAMES = ["scaler_btc.gz", "scaler_btc_return.gz", "model_metadata_btc.json"]


def _create_s3_manager() -> S3ModelManager:
    """Cria instância de S3ModelManager para o bucket de modelos.

    Raises:
        RuntimeError: Se S3_MODELS_BUCKET não estiver configurado.
    """
    if not S3_MODELS_BUCKET:
        raise RuntimeError("S3_MODELS_BUCKET não definido; não é possível carregar modelos do S3.")
    return S3ModelManager(bucket_name=S3_MODELS_BUCKET)


def load_challenger_from_s3(s3_manager: S3ModelManager) -> Any:
    """Carrega o modelo challenger diretamente do S3.

    Args:
        s3_manager: Instância de S3ModelManager.

    Returns:
        Modelo Keras carregado.

    Raises:
        RuntimeError: Se o challenger não for encontrado em S3.
    """
    path = f"{S3_CHALLENGER_PREFIX}/{S3_MODEL_FILENAME}"
    try:
        model = s3_manager.load_model(path)
        logger.info("Challenger carregado de S3: %s/%s", S3_MODELS_BUCKET, path)
        return model
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Challenger não encontrado em S3: {S3_MODELS_BUCKET}/models/{path}"
        ) from exc


def load_champion_from_s3(s3_manager: S3ModelManager) -> Any | None:
    """Carrega o modelo champion diretamente do S3.

    Args:
        s3_manager: Instância de S3ModelManager.

    Returns:
        Modelo Keras carregado, ou None se ainda não houver champion em produção.
    """
    path = f"{S3_CHAMPION_PREFIX}/{S3_MODEL_FILENAME}"
    try:
        model = s3_manager.load_model(path)
        logger.info("Champion carregado de S3: %s/%s", S3_MODELS_BUCKET, path)
        return model
    except FileNotFoundError:
        logger.info(
            "Nenhum champion encontrado em S3 (%s/models/%s); auto-promoção.",
            S3_MODELS_BUCKET,
            path,
        )
        return None


def promote_artifacts_s3(s3_manager: S3ModelManager) -> None:
    """Copia artefatos do prefixo challenger para champion no mesmo bucket S3.

    Args:
        s3_manager: Instância de S3ModelManager.
    """
    files = [S3_MODEL_FILENAME, *S3_SCALER_FILENAMES]
    promoted: list[str] = []
    for filename in files:
        src_key = s3_manager._s3_key(f"{S3_CHALLENGER_PREFIX}/{filename}")
        dst_key = s3_manager._s3_key(f"{S3_CHAMPION_PREFIX}/{filename}")
        try:
            s3_manager.copy_object(src_key, dst_key)
            promoted.append(filename)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Falha ao copiar %s para champion (ignorado): %s", filename, exc)
    logger.info(
        "Promoção S3 concluída: %d/%d arquivos copiados de '%s' para '%s'.",
        len(promoted),
        len(files),
        S3_CHALLENGER_PREFIX,
        S3_CHAMPION_PREFIX,
    )


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
    aliases_to_try: list[str] = [CANDIDATE_ALIAS, *FALLBACK_CANDIDATE_ALIASES]
    # Remove duplicatas preservando ordem
    aliases_to_try = list(dict.fromkeys(aliases_to_try))

    model_version = None
    used_alias = ""
    last_error: Exception | None = None
    for alias in aliases_to_try:
        try:
            model_version = client.get_model_version_by_alias(MLFLOW_MODEL_NAME, alias)
            used_alias = alias
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    if model_version is None:
        raise RuntimeError(
            "Nenhum alias de challenger encontrado para "
            f"'{MLFLOW_MODEL_NAME}'. Tentados: {aliases_to_try}. Último erro: {last_error}"
        ) from last_error

    version = str(model_version.version)
    run_id = str(model_version.run_id)
    logger.info(
        "Challenger identificado via alias '%s': version=%s run_id=%s",
        used_alias,
        version,
        run_id,
    )
    return version, run_id


def _load_holdout_from_s3(s3_manager: Any) -> tuple[np.ndarray, np.ndarray]:
    """Carrega holdout (X_test, y_test) salvo pelo treinamento a partir do S3.

    Args:
        s3_manager: Instância de S3ModelManager.

    Returns:
        Tupla ``(X_holdout, y_holdout)`` como arrays numpy.

    Raises:
        RuntimeError: Se o holdout não for encontrado em S3.
    """
    try:
        holdout = s3_manager.load_joblib("challenger/holdout.joblib")
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Holdout não encontrado em S3 (challenger/holdout.joblib). "
            "Execute o treinamento novamente para gerar o artefato."
        ) from exc

    X_holdout: np.ndarray = holdout["X"]
    y_holdout: np.ndarray = holdout["y"]
    return X_holdout, y_holdout


def _predict_scores(model: Any, X_holdout: Any) -> np.ndarray:
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


def _is_truthy_tag(value: str | None) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _validate_challenger_lineage(client: Any, challenger_run_id: str) -> tuple[bool, str]:
    """Valida se o challenger possui lineage completo para ser elegível à promoção."""
    run = client.get_run(challenger_run_id)
    run_tags = getattr(getattr(run, "data", None), "tags", {}) or {}

    lineage_complete = _is_truthy_tag(run_tags.get("lineage_complete"))
    git_sha = str(run_tags.get("git_sha", "")).strip().lower()

    if lineage_complete and git_sha and git_sha != "unknown":
        return True, ""

    reason = (
        "lineage_incomplete "
        f"lineage_complete={run_tags.get('lineage_complete', '<missing>')} "
        f"git_sha={run_tags.get('git_sha', '<missing>')}"
    )
    return False, reason


def main() -> int:
    """Executa validação Champion-Challenger baseada em AUC + quality gate.

    Returns:
        Exit code para integração com Step Functions.
    """
    try:
        mlflow_module = _configure_mlflow()
        client = mlflow_module.MlflowClient()

        challenger_version, challenger_run_id = _resolve_candidate_version(client)
        lineage_ok, lineage_reason = _validate_challenger_lineage(client, challenger_run_id)
        if not lineage_ok:
            _mark_rejected(client, challenger_version, lineage_reason)
            client.set_model_version_tag(
                name=MLFLOW_MODEL_NAME,
                version=challenger_version,
                key="promotion_gate",
                value="lineage_incomplete",
            )
            logger.info(
                "REJECTED: challenger not promoted due to incomplete lineage | "
                "challenger_version=%s reason=%s",
                challenger_version,
                lineage_reason,
            )
            return EXIT_NOT_PROMOTED

        s3_manager = _create_s3_manager()

        # Carrega modelos diretamente do S3 (fonte de verdade)
        challenger_model = load_challenger_from_s3(s3_manager)
        champion_model = load_champion_from_s3(s3_manager)  # None se não há champion ainda

        if champion_model is None:
            # Sem champion em produção: auto-promoção do primeiro modelo sem gate de AUC.
            promote_artifacts_s3(s3_manager)
            _promote_to_production_alias(client, challenger_version, float("inf"))
            logger.info(
                "APPROVED: primeiro modelo auto-promovido a champion "
                "(S3 + MLflow alias '%s') | challenger_version=%s",
                PRODUCTION_ALIAS,
                challenger_version,
            )
            return EXIT_PROMOTED

        # Champion existe: carrega holdout do S3 e compara AUC.
        X_holdout, y_holdout = _load_holdout_from_s3(s3_manager)

        # O modelo é regressão (log_return contínuo). Binariza para AUC:
        # retorno > 0 → 1 (preço subiu), ≤ 0 → 0 (preço caiu/estável).
        y_binary = (y_holdout > 0).astype(int)
        logger.info(
            "Holdout binarizado para AUC | amostras=%d positivos=%d (%.1f%%)",
            len(y_binary),
            int(y_binary.sum()),
            100.0 * y_binary.mean(),
        )

        challenger_scores = _predict_scores(challenger_model, X_holdout)
        challenger_auc = float(roc_auc_score(y_binary, challenger_scores))
        champion_scores = _predict_scores(champion_model, X_holdout)
        champion_auc = float(roc_auc_score(y_binary, champion_scores))
        delta_auc = challenger_auc - champion_auc

        logger.info(
            "Evaluation summary | champion_auc=%.6f challenger_auc=%.6f"
            " delta_auc=%.6f threshold=%.6f",
            champion_auc,
            challenger_auc,
            delta_auc,
            MIN_IMPROVEMENT,
        )

        if delta_auc >= MIN_IMPROVEMENT:
            # 1. Copia artefatos challenger → champion no S3
            promote_artifacts_s3(s3_manager)
            # 2. Atualiza alias no MLflow Registry para governança
            _promote_to_production_alias(client, challenger_version, float(delta_auc))
            logger.info(
                "APPROVED: challenger promoted to champion "
                "(S3 + MLflow alias '%s') | challenger_version=%s",
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
