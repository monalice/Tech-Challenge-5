import logging
from typing import Any

import psutil
from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

import src.delivery.api.dependencies as _deps
from src.adapters.observability.prometheus import METRIC_CPU, METRIC_MEMORY
from src.delivery.api.schemas import HealthResponse, LiveResponse
from src.domain.ports import LoadedArtifacts
from src.use_cases.health_check import perform_health_checks

logger = logging.getLogger("stockcast")

router = APIRouter()


@router.get(
    "/live",
    response_model=LiveResponse,
    summary="Liveness da API",
    description="Endpoint leve para healthcheck de container, sem consulta externa.",
)
def live_check(request: Request) -> dict[str, Any]:
    _artifacts: LoadedArtifacts | None = getattr(request.app.state, "artifacts", None)
    return {"status": "alive", "artifacts_ready": _artifacts is not None}


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Saúde efetiva da API",
    description=(
        "Valida artefatos, inferência do modelo e acesso ao mercado. "
        "Indica a fonte de dados ativa (yfinance ou binance). "
        "Retorna timestamps em UTC e Brasília."
    ),
)
def health_check(request: Request) -> dict[str, Any]:
    _artifacts: LoadedArtifacts | None = getattr(request.app.state, "artifacts", None)
    return perform_health_checks(
        _artifacts,
        download_market_data=_deps.download_with_retry,
        metric_cpu=METRIC_CPU,
        metric_memory=METRIC_MEMORY,
    )


@router.get(
    "/metrics",
    response_class=PlainTextResponse,
    summary="Métricas Prometheus",
    description=(
        "Expõe métricas operacionais no formato Prometheus/OpenMetrics para coleta "
        "por ferramentas de monitoramento "
        "(Prometheus, Grafana, etc.). Inclui contadores de requisições, latência de inferência, "
        "uso de fontes de dados e métricas de recursos do sistema."
    ),
)
def prometheus_metrics() -> PlainTextResponse:
    METRIC_CPU.set(psutil.cpu_percent())
    METRIC_MEMORY.set(psutil.virtual_memory().percent)
    return PlainTextResponse(
        content=generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )
