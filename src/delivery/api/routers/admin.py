from typing import Any

import mlflow
from fastapi import APIRouter, Body

import src.delivery.api.dependencies as _deps
from src.delivery.api.schemas import DriftCheckRequest
from src.domain.drift.detection import detect_data_drift
from src.use_cases.drift_check import DriftAutomationConfig, process_drift_result

router = APIRouter(prefix="/admin")


@router.post(
    "/check-drift",
    summary="Executa checagem de data drift",
    description=(
        "Executa a deteccao de data drift via PSI comparando historico de previsoes e dados reais. "
        "Endpoint administrativo para automacao MLOps."
    ),
)
async def check_drift(
    request: DriftCheckRequest = Body(default_factory=DriftCheckRequest),  # noqa: B008
) -> dict[str, Any]:
    result = await detect_data_drift(
        ticker=request.ticker.upper(),
        download_fn=_deps.download_with_retry,
        prediction_log=_deps.prediction_log,
    )

    automation_summary = process_drift_result(
        result,
        DriftAutomationConfig.from_sources(),
        mlflow_module=mlflow,
    )

    return {**result, "automation": automation_summary}
