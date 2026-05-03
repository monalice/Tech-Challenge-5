import logging
import time
from typing import Any

import pandas as pd
from fastapi import APIRouter, Body, HTTPException, Request

from src.adapters.observability.prometheus import (
    METRIC_PREDICT_LATENCY,
    METRIC_PREDICT_REQUESTS,
)
from src.delivery.api.dependencies import PREDICTIONS_HISTORY_MAX, prediction_log
from src.delivery.api.schemas import (
    ConfidenceIntervalResponse,
    CryptoRequest,
    PredictionHistoryResponse,
    PredictionResponse,
)
from src.domain.constants import SUPPORTED_TICKER
from src.domain.inference import DataServiceError, InferenceService, InsufficientDataError
from src.domain.time_utils import timestamp_to_brt_iso, timestamp_to_utc_iso

logger = logging.getLogger("stockcast")

router = APIRouter()


@router.get(
    "/predictions/history",
    response_model=PredictionHistoryResponse,
    summary="Histórico de previsões",
    description=(
        f"Retorna as últimas até {PREDICTIONS_HISTORY_MAX} previsões realizadas "
        "pelo endpoint /predict em ordem decrescente (mais recente primeiro). "
        "Útil para auditoria, monitoramento de drift e "
        "comparação de previsões com valores reais."
    ),
)
def predictions_history() -> dict[str, Any]:
    entries = list(reversed(prediction_log))
    return {"total_logged": len(entries), "predictions": entries}


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Prevê o próximo fechamento horário",
    description=(
        "Aceita apenas o ticker BTC-USD. "
        "O body é opcional: você pode omitir o body ou enviar {} "
        "para usar o padrão BTC-USD. "
        "Por padrão usa apenas velas fechadas; para incluir a vela em formação, "
        "use use_partial_candle=true. "
        "Retorna preço previsto, janela temporal da previsão em UTC/Brasília, "
        "intervalo de confiança, erro estimado e a fonte de dados utilizada."
    ),
)
def predict_next_hour(
    http_request: Request,
    request: CryptoRequest = Body(  # noqa: B008
        default_factory=CryptoRequest,
        openapi_examples={
            "sem_body_ou_vazio": {
                "summary": "Sem body ou body vazio",
                "description": "Pode omitir o body ou enviar {}. O ticker padrão será BTC-USD.",
                "value": {},
            },
            "body_explicito": {"summary": "Body explícito", "value": {"ticker": "BTC-USD"}},
            "com_vela_parcial": {
                "summary": "Com vela parcial",
                "description": "Inclui a vela horária em formação na entrada do modelo.",
                "value": {"ticker": "BTC-USD", "use_partial_candle": True},
            },
        },
    ),
) -> dict[str, Any]:
    start_proc = time.perf_counter()
    ticker = request.ticker.upper()

    if ticker != SUPPORTED_TICKER:
        METRIC_PREDICT_REQUESTS.labels(ticker=ticker, status="error_unsupported").inc()
        raise HTTPException(
            status_code=400, detail=f"Este modelo foi treinado apenas para {SUPPORTED_TICKER}."
        )

    service: InferenceService | None = getattr(http_request.app.state, "service", None)
    if service is None:
        METRIC_PREDICT_REQUESTS.labels(ticker=ticker, status="error_no_model").inc()
        raise HTTPException(status_code=503, detail="Modelo não disponível.")

    try:
        result = service.predict(ticker, request.use_partial_candle)
    except DataServiceError as exc:
        METRIC_PREDICT_REQUESTS.labels(ticker=ticker, status="error").inc()
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except InsufficientDataError as exc:
        METRIC_PREDICT_REQUESTS.labels(ticker=ticker, status="error").inc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        METRIC_PREDICT_REQUESTS.labels(ticker=ticker, status="error").inc()
        raise
    except Exception as exc:
        METRIC_PREDICT_REQUESTS.labels(ticker=ticker, status="error_internal").inc()
        logger.error("Erro interno em /predict: %s: %s", type(exc).__name__, exc)
        raise HTTPException(status_code=500, detail="Falha interna ao gerar previsão") from None

    forecast_for_ts = result.last_observed_ts + pd.Timedelta(hours=1)
    forecast_close_ts = forecast_for_ts + pd.Timedelta(hours=1) - pd.Timedelta(seconds=1)
    input_mode = "include_partial_candle" if request.use_partial_candle else "closed_candles_only"
    confidence_interval_95: ConfidenceIntervalResponse | None = (
        ConfidenceIntervalResponse(
            low_usd=result.confidence_interval.low_usd,
            high_usd=result.confidence_interval.high_usd,
        )
        if result.confidence_interval is not None
        else None
    )

    proc_time = (time.perf_counter() - start_proc) * 1000
    METRIC_PREDICT_LATENCY.observe(proc_time / 1000)
    METRIC_PREDICT_REQUESTS.labels(ticker=ticker, status="success").inc()

    prediction_log.append(
        {
            "requested_at_utc": pd.Timestamp.utcnow().isoformat(),
            "ticker": ticker,
            "input_mode": input_mode,
            "last_input_candle_utc": timestamp_to_utc_iso(result.last_observed_ts),
            "forecast_for_utc": timestamp_to_utc_iso(forecast_for_ts),
            "predicted_price_usd": round(result.predicted_price_usd, 2),
            "data_source": result.data_source,
            "processing_time_ms": round(proc_time, 2),
        }
    )
    logger.info(
        "Previsão gerada | ticker=%s source=%s price=%.2f latency=%.1fms",
        ticker,
        result.data_source,
        result.predicted_price_usd,
        proc_time,
    )

    return {
        "ticker": ticker,
        "prediction_type": "Next Hour Close",
        "input_mode": input_mode,
        "last_input_candle_utc": timestamp_to_utc_iso(result.last_observed_ts),
        "last_input_candle_brt": timestamp_to_brt_iso(result.last_observed_ts),
        "predicted_price_usd": round(result.predicted_price_usd, 2),
        "forecast_for_utc": timestamp_to_utc_iso(forecast_for_ts),
        "forecast_for_brt": timestamp_to_brt_iso(forecast_for_ts),
        "forecast_close_utc": timestamp_to_utc_iso(forecast_close_ts),
        "forecast_close_brt": timestamp_to_brt_iso(forecast_close_ts),
        "confidence_interval_95_usd": confidence_interval_95,
        "estimated_error_pct": (
            None
            if result.estimated_error_pct is None
            else round(float(result.estimated_error_pct), 2)
        ),
        "data_source": result.data_source,
        "processing_time_ms": round(proc_time, 2),
    }
