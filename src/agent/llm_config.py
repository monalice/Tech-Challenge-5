"""Configuração e validação de LLM para Amazon Bedrock.

Centraliza constantes, validação de ambiente e publicação de métricas
CloudWatch relacionadas ao ciclo de vida do agente LLM. Extraído de
``src/app.py`` seguindo os princípios de Clean Architecture para que o
entrypoint FastAPI não carregue responsabilidades de domínio.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("stockcast.llm_config")

# ---------------------------------------------------------------------------
# Constantes de validação de ambiente Bedrock
# ---------------------------------------------------------------------------

BEDROCK_REGION_PATTERN: re.Pattern[str] = re.compile(r"^[a-z]{2}(?:-gov)?-[a-z]+-\d$")

INSECURE_BEDROCK_CONFIG_VALUES: frozenset[str] = frozenset(
    {
        "",
        "your-aws-region",
        "your-bedrock-region",
        "changeme",
        "replace-me",
        "your_region_here",
        "test",
        "dummy",
        "none",
        "null",
    }
)

#: Nomes de ambiente considerados produção (case-insensitive após strip).
PRODUCTION_ENV_NAMES: frozenset[str] = frozenset({"prod", "production"})

#: Variáveis de ambiente inspecionadas para detectar o ambiente de execução.
PRODUCTION_ENV_VARIABLES: tuple[str, ...] = (
    "APP_ENV",
    "ENVIRONMENT",
    "ENV",
    "STAGE",
    "DEPLOY_ENV",
)

#: Namespace padrão CloudWatch para métricas LLM.
DEFAULT_CLOUDWATCH_NAMESPACE: str = "StockCast/LLM"

#: Dimensões publicadas nas métricas CloudWatch do LLM.
CLOUDWATCH_METRIC_DIMENSIONS: list[str] = ["Service", "Environment", "Endpoint"]


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------


def _is_true(value: str | None, default: bool = True) -> bool:
    """Converte uma variável de ambiente em booleano.

    Args:
        value: String lida de uma variável de ambiente, ou ``None`` quando ausente.
        default: Valor retornado quando *value* é ``None``.

    Returns:
        ``True`` se *value* for uma das strings canônicas de verdade
        (``"1"``, ``"true"``, ``"yes"``, ``"on"``); caso contrário ``False``.
        Retorna *default* quando *value* é ``None``.
    """
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_cloudwatch_region() -> str | None:
    """Resolve a região AWS para publicação de métricas CloudWatch.

    Lê as variáveis de ambiente em ordem de prioridade: ``BEDROCK_AWS_REGION``,
    ``AWS_REGION``, ``AWS_DEFAULT_REGION``.

    Returns:
        Região AWS como string (ex: ``"us-east-1"``), ou ``None`` se nenhuma
        variável estiver definida ou não-vazia.
    """
    for env_name in ("BEDROCK_AWS_REGION", "AWS_REGION", "AWS_DEFAULT_REGION"):
        candidate = os.getenv(env_name)
        if candidate and candidate.strip():
            return candidate.strip()
    return None


# ---------------------------------------------------------------------------
# Funções públicas do domínio LLM/Bedrock
# ---------------------------------------------------------------------------


def is_production_environment() -> bool:
    """Verifica se a aplicação está rodando em ambiente de produção.

    Inspeciona as variáveis de ambiente listadas em :data:`PRODUCTION_ENV_VARIABLES`
    e retorna ``True`` se qualquer uma delas contiver um valor em
    :data:`PRODUCTION_ENV_NAMES`.

    Returns:
        ``True`` quando pelo menos uma variável de ambiente indica produção;
        ``False`` caso contrário.
    """
    for env_var in PRODUCTION_ENV_VARIABLES:
        value = os.getenv(env_var)
        if value and value.strip().lower() in PRODUCTION_ENV_NAMES:
            return True
    return False


def validate_bedrock_configuration_for_startup() -> None:
    """Valida a configuração do Amazon Bedrock antes de subir a API em produção.

    Em ambientes não-produtivos, retorna imediatamente sem realizar nenhuma
    verificação. Em produção, garante que:

    * A região AWS seja uma string real (não placeholder) com formato válido.
    * ``BEDROCK_GUARDRAIL_ID`` e ``BEDROCK_GUARDRAIL_VERSION`` estejam definidos.

    Raises:
        RuntimeError: Se qualquer condição de segurança não for satisfeita em produção.
    """
    if not is_production_environment():
        return

    aws_region = (
        os.getenv("BEDROCK_AWS_REGION")
        or os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
        or ""
    ).strip()

    if aws_region.lower() in INSECURE_BEDROCK_CONFIG_VALUES:
        raise RuntimeError(
            "Região AWS do Amazon Bedrock inválida para produção. "
            "Configure uma região real e segura."
        )

    if not BEDROCK_REGION_PATTERN.fullmatch(aws_region):
        raise RuntimeError(
            "Região AWS do Amazon Bedrock com formato inválido para produção. "
            "Use uma região válida antes de iniciar a API."
        )

    guardrail_identifier = (os.getenv("BEDROCK_GUARDRAIL_ID") or "").strip()
    guardrail_version = (os.getenv("BEDROCK_GUARDRAIL_VERSION") or "").strip()
    if not guardrail_identifier or not guardrail_version:
        raise RuntimeError(
            "Amazon Bedrock Guardrails deve ser configurado em produção. "
            "Defina BEDROCK_GUARDRAIL_ID e BEDROCK_GUARDRAIL_VERSION."
        )


def publish_cloudwatch_llm_metrics(*, latency_ms: float, is_error: bool) -> None:
    """Publica métricas de latência e taxa de erro do LLM no Amazon CloudWatch.

    A publicação é suprimida silenciosamente quando ``CW_LLM_METRICS_ENABLED``
    for falso ou quando ``boto3`` não estiver disponível no ambiente.

    Args:
        latency_ms: Latência da chamada ao agente LLM em milissegundos.
        is_error: ``True`` se a chamada resultou em erro; ``False`` caso contrário.
    """
    if not _is_true(os.getenv("CW_LLM_METRICS_ENABLED"), default=True):
        return

    try:
        import boto3  # type: ignore[import-untyped]
    except Exception as exc:
        logger.warning("boto3 indisponível para métricas CloudWatch: %s", exc)
        return

    region_name = _resolve_cloudwatch_region()
    namespace = os.getenv("CW_LLM_METRICS_NAMESPACE", DEFAULT_CLOUDWATCH_NAMESPACE)
    service_name = os.getenv("CW_METRIC_SERVICE_NAME", "stockcast")
    environment_name = os.getenv("CW_METRIC_ENVIRONMENT", os.getenv("APP_ENV", "unknown"))
    dimensions: list[dict[str, str]] = [
        {"Name": "Service", "Value": service_name},
        {"Name": "Environment", "Value": environment_name},
        {"Name": "Endpoint", "Value": "/chat"},
    ]

    now = datetime.now(timezone.utc)
    metric_data: list[dict[str, Any]] = [
        {
            "MetricName": "llm_latency",
            "Dimensions": dimensions,
            "Timestamp": now,
            "Value": float(latency_ms),
            "Unit": "Milliseconds",
        },
        {
            "MetricName": "llm_error_rate",
            "Dimensions": dimensions,
            "Timestamp": now,
            "Value": 100.0 if is_error else 0.0,
            "Unit": "Percent",
        },
    ]

    try:
        cloudwatch = boto3.client("cloudwatch", region_name=region_name)
        cloudwatch.put_metric_data(Namespace=namespace, MetricData=metric_data)
    except Exception as exc:
        logger.error("Falha ao publicar métricas LLM no CloudWatch: %s", exc)
