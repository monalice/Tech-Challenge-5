"""Avaliação e promoção do modelo Challenger vs Champion no MLflow Model Registry.

Executado como task ECS dedicada dentro do pipeline Step Functions de
champion-challenger. Nunca deve ser chamado diretamente pelo serviço de inferência.

Fluxo:
    1. Localiza a versão mais recente do modelo com alias 'candidate' no MLflow.
    2. Lê a métrica ``mae_price`` do run associado ao challenger.
    3. Lê a métrica ``mae_price`` do run associado ao champion atual.
    4. Calcula a melhoria relativa: (champion_mae - challenger_mae) / champion_mae.
    5. Se melhoria >= CHAMPION_MIN_IMPROVEMENT:
       - Promove challenger → alias 'champion' (arquiva champion anterior).
       - Sai com código 0.
    6. Se melhoria < CHAMPION_MIN_IMPROVEMENT:
       - Mantém challenger como 'candidate' com tag de motivo.
       - Sai com código 10 (Step Functions trata como ChallengerNotPromoted).
    7. Em caso de falha de infraestrutura/API: sai com código 1 (TrainingFailed).

Variáveis de ambiente:
    MLFLOW_TRACKING_URI         URI do servidor MLflow (obrigatório).
    MLFLOW_MODEL_NAME           Nome do modelo registrado (default: btc_hourly_forecaster).
    MLFLOW_CHAMPION_ALIAS       Alias do champion em produção (default: champion).
    MLFLOW_CANDIDATE_ALIAS      Alias do challenger avaliado (default: candidate).
    CHAMPION_MIN_IMPROVEMENT    Melhoria mínima relativa de MAE (default: 0.005 = 0.5%).
    CHAMPION_METRIC             Métrica usada na comparação (default: mae_price).
"""

from __future__ import annotations

import logging
import os
import sys

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
EXIT_NOT_PROMOTED = 10   # challenger below threshold — not an error
EXIT_SYSTEM_ERROR = 1    # infrastructure/API failure

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "btc_hourly_forecaster")
CHAMPION_ALIAS = os.getenv("MLFLOW_CHAMPION_ALIAS", "champion")
CANDIDATE_ALIAS = os.getenv("MLFLOW_CANDIDATE_ALIAS", "candidate")
CHAMPION_METRIC = os.getenv("CHAMPION_METRIC", "mae_price")
_min_improvement_raw = os.getenv("CHAMPION_MIN_IMPROVEMENT", "0.005")
MIN_IMPROVEMENT: float = float(_min_improvement_raw)


def _configure_mlflow() -> None:
    """Configura o tracking URI do MLflow a partir da variável de ambiente.

    Raises:
        RuntimeError: Se MLFLOW_TRACKING_URI não estiver definido.
    """
    import mlflow  # type: ignore[import-untyped]

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if not tracking_uri:
        raise RuntimeError(
            "MLFLOW_TRACKING_URI não está definido. "
            "Defina a variável de ambiente antes de executar este script."
        )
    mlflow.set_tracking_uri(tracking_uri)
    logger.info("MLflow tracking URI: %s", tracking_uri)


def _resolve_candidate_version() -> tuple[str, str]:
    """Retorna (version_number, run_id) da versão com alias 'candidate'.

    Returns:
        Tupla (version, run_id) do modelo challenger candidato.

    Raises:
        RuntimeError: Se nenhuma versão com o alias candidate for encontrada.
    """
    import mlflow

    client = mlflow.MlflowClient()
    try:
        model_version = client.get_model_version_by_alias(MLFLOW_MODEL_NAME, CANDIDATE_ALIAS)
    except Exception as exc:
        raise RuntimeError(
            f"Nenhum modelo com alias '{CANDIDATE_ALIAS}' encontrado em '{MLFLOW_MODEL_NAME}'. "
            f"Certifique-se de que o treinamento registrou um candidato. Erro: {exc}"
        ) from exc

    logger.info(
        "Challenger (candidate) identificado: versão %s, run_id=%s",
        model_version.version,
        model_version.run_id,
    )
    return str(model_version.version), str(model_version.run_id)


def _get_metric(run_id: str, metric_name: str) -> float:
    """Lê uma métrica de um run MLflow.

    Args:
        run_id: ID do run no MLflow.
        metric_name: Nome da métrica a ser lida.

    Returns:
        Valor da métrica como float.

    Raises:
        KeyError: Se a métrica não existir no run.
    """
    import mlflow

    client = mlflow.MlflowClient()
    run = client.get_run(run_id)
    value = run.data.metrics.get(metric_name)
    if value is None:
        raise KeyError(
            f"Métrica '{metric_name}' não encontrada no run '{run_id}'. "
            f"Métricas disponíveis: {list(run.data.metrics.keys())}"
        )
    return float(value)


def _resolve_champion_mae() -> tuple[float, str | None]:
    """Lê o MAE do champion atual. Retorna (mae, run_id) ou (inf, None) se não houver champion.

    Returns:
        Tupla (mae, run_id): mae=inf e run_id=None quando não há champion definido.
    """
    import mlflow

    client = mlflow.MlflowClient()
    try:
        champion_version = client.get_model_version_by_alias(MLFLOW_MODEL_NAME, CHAMPION_ALIAS)
    except Exception:
        logger.info(
            "Nenhum champion com alias '%s' encontrado. Challenger será promovido automaticamente.",
            CHAMPION_ALIAS,
        )
        return float("inf"), None

    mae = _get_metric(champion_version.run_id, CHAMPION_METRIC)
    logger.info(
        "Champion (alias=%s): versão %s, %s=%.6f",
        CHAMPION_ALIAS,
        champion_version.version,
        CHAMPION_METRIC,
        mae,
    )
    return mae, champion_version.run_id


def _promote_challenger(challenger_version: str) -> None:
    """Promove challenger via alias champion e arquiva o champion anterior.

    Args:
        challenger_version: Número da versão do modelo challenger a promover.
    """
    import mlflow

    client = mlflow.MlflowClient()

    # Arquiva o champion anterior se existir
    try:
        previous = client.get_model_version_by_alias(MLFLOW_MODEL_NAME, CHAMPION_ALIAS)
        if str(previous.version) != str(challenger_version):
            client.set_model_version_tag(
                name=MLFLOW_MODEL_NAME,
                version=previous.version,
                key="lifecycle_state",
                value="archived",
            )
            client.set_model_version_tag(
                name=MLFLOW_MODEL_NAME,
                version=previous.version,
                key="champion_replaced_by",
                value=str(challenger_version),
            )
            logger.info("Champion anterior v%s marcado como 'archived'.", previous.version)
    except Exception:
        pass  # sem champion anterior — promoção direta

    client.set_registered_model_alias(
        name=MLFLOW_MODEL_NAME,
        alias=CHAMPION_ALIAS,
        version=challenger_version,
    )
    client.set_model_version_tag(
        name=MLFLOW_MODEL_NAME,
        version=challenger_version,
        key="lifecycle_state",
        value="champion",
    )
    client.set_model_version_tag(
        name=MLFLOW_MODEL_NAME,
        version=challenger_version,
        key="promotion_trigger",
        value="step_functions_champion_challenger",
    )
    logger.info("Challenger v%s promovido ao alias '%s'.", challenger_version, CHAMPION_ALIAS)


def _keep_as_candidate(challenger_version: str, reason: str) -> None:
    """Mantém challenger como candidato com tag de motivo.

    Args:
        challenger_version: Número da versão do modelo challenger.
        reason: Motivo pelo qual o challenger não foi promovido.
    """
    import mlflow

    client = mlflow.MlflowClient()
    client.set_model_version_tag(
        name=MLFLOW_MODEL_NAME,
        version=challenger_version,
        key="lifecycle_state",
        value="candidate",
    )
    client.set_model_version_tag(
        name=MLFLOW_MODEL_NAME,
        version=challenger_version,
        key="promotion_gate",
        value="metric_gate_not_passed",
    )
    client.set_model_version_tag(
        name=MLFLOW_MODEL_NAME,
        version=challenger_version,
        key="candidate_reason",
        value=reason,
    )
    logger.info(
        "Challenger v%s mantido como candidato. Motivo: %s", challenger_version, reason
    )


def main() -> int:
    """Ponto de entrada: avalia challenger vs champion e promove ou arquiva.

    Returns:
        Código de saída:
            0  — challenger promovido com sucesso.
            10 — challenger abaixo do threshold; champion mantido.
            1  — erro de infraestrutura/API.
    """
    try:
        _configure_mlflow()
    except RuntimeError as exc:
        logger.error("Configuração MLflow falhou: %s", exc)
        return EXIT_SYSTEM_ERROR

    try:
        challenger_version, challenger_run_id = _resolve_candidate_version()
    except RuntimeError as exc:
        logger.error("Challenger não encontrado: %s", exc)
        return EXIT_SYSTEM_ERROR

    try:
        challenger_mae = _get_metric(challenger_run_id, CHAMPION_METRIC)
        logger.info("Challenger v%s %s=%.6f", challenger_version, CHAMPION_METRIC, challenger_mae)
    except (KeyError, Exception) as exc:
        logger.error("Falha ao ler métrica do challenger: %s", exc)
        return EXIT_SYSTEM_ERROR

    try:
        champion_mae, _ = _resolve_champion_mae()
    except Exception as exc:
        logger.error("Falha ao ler métricas do champion: %s", exc)
        return EXIT_SYSTEM_ERROR

    if champion_mae == float("inf"):
        # Sem champion — primeira promoção
        improvement = float("inf")
    else:
        improvement = (champion_mae - challenger_mae) / champion_mae

    logger.info(
        "Comparação: champion_%s=%.6f | challenger_%s=%.6f | melhoria=%.4f%% | threshold=%.4f%%",
        CHAMPION_METRIC,
        champion_mae,
        CHAMPION_METRIC,
        challenger_mae,
        improvement * 100,
        MIN_IMPROVEMENT * 100,
    )

    if improvement >= MIN_IMPROVEMENT:
        try:
            _promote_challenger(challenger_version)
            logger.info(
                "Challenger v%s PROMOVIDO. Melhoria de %.4f%% >= threshold de %.4f%%.",
                challenger_version,
                improvement * 100,
                MIN_IMPROVEMENT * 100,
            )
            return EXIT_PROMOTED
        except Exception as exc:
            logger.error("Falha ao promover challenger v%s: %s", challenger_version, exc)
            return EXIT_SYSTEM_ERROR
    else:
        reason = (
            f"improvement={improvement:.6f} < threshold={MIN_IMPROVEMENT} "
            f"(champion_mae={champion_mae:.6f}, challenger_mae={challenger_mae:.6f})"
        )
        try:
            _keep_as_candidate(challenger_version, reason)
        except Exception as exc:
            logger.warning("Falha ao registrar motivo no MLflow: %s", exc)

        logger.info(
            "Champion mantido. Challenger v%s abaixo do threshold (%.4f%% < %.4f%%).",
            challenger_version,
            improvement * 100,
            MIN_IMPROVEMENT * 100,
        )
        return EXIT_NOT_PROMOTED


if __name__ == "__main__":
    sys.exit(main())
