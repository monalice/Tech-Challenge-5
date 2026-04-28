import collections
import json
import logging
import os
import re
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from zoneinfo import ZoneInfo

import joblib
import mlflow
import numpy as np
import pandas as pd
import psutil
import requests

# IMPORTANTE (Windows): tensorflow e yfinance devem ser importados antes de pandas
# para evitar conflito de DLL que causa crash (exit code -1073741819)
import tensorflow as tf
import uvicorn
import yfinance as yf
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, ConfigDict, Field

from monitoring.drift_detection import detect_data_drift
from src.features.technical_features import (
    build_feature_matrix as _build_feature_matrix,
)
from src.features.technical_features import (
    compute_bollinger_pct_b as _compute_bollinger_pct_b,  # noqa: F401
)
from src.features.technical_features import (
    compute_macd_signal as _compute_macd_signal,  # noqa: F401
)
from src.features.technical_features import (
    compute_rsi as _compute_rsi,  # noqa: F401
)
from src.features.technical_features import (
    compute_sma_ratio as _compute_sma_ratio,  # noqa: F401
)
from src.features.technical_features import (
    compute_volume_ratio as _compute_volume_ratio,  # noqa: F401
)
from src.security.guardrails import InputGuardrail, OutputGuardrail
from src.serving.drift_automation import DriftAutomationConfig, process_drift_result

# --- Logging estruturado ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger("stockcast")

input_guardrail = InputGuardrail()
output_guardrail = OutputGuardrail()

PRODUCTION_ENV_NAMES = {"prod", "production"}
PRODUCTION_ENV_VARIABLES = (
    "APP_ENV",
    "ENVIRONMENT",
    "ENV",
    "STAGE",
    "DEPLOY_ENV",
)
GOOGLE_API_KEY_PATTERN = re.compile(r"^AIza[A-Za-z0-9_-]{20,}$")
INSECURE_GOOGLE_API_KEY_VALUES = {
    "",
    "your-google-api-key",
    "changeme",
    "replace-me",
    "your_api_key_here",
    "mock_key_para_testes",
    "test",
    "dummy",
    "none",
    "null",
}


# --- Schemas ---
class CryptoRequest(BaseModel):
    ticker: str = Field(
        default="BTC-USD",
        description="Ticker do criptoativo. Apenas BTC-USD é suportado",
    )
    use_partial_candle: bool = Field(
        default=False,
        description=(
            "Se true, usa também a vela horária em formação. "
            "Se false (padrão), usa apenas velas fechadas"
        ),
    )


class ConfidenceIntervalResponse(BaseModel):
    low_usd: float = Field(description="Limite inferior em USD")
    high_usd: float = Field(description="Limite superior em USD")


class PredictionResponse(BaseModel):
    ticker: str = Field(description="Ticker previsto")
    prediction_type: str = Field(description="Tipo de previsão")
    input_mode: str = Field(
        description=(
            "Modo de entrada usado: closed_candles_only "
            "ou include_partial_candle"
        )
    )
    last_input_candle_utc: str = Field(
        description="Último candle usado como entrada em UTC (ISO-8601)"
    )
    last_input_candle_brt: str = Field(
        description="Último candle usado como entrada em Brasília (ISO-8601)"
    )
    predicted_price_usd: float = Field(
        description="Preço previsto para o fechamento da próxima hora"
    )
    forecast_for_utc: str = Field(description="Início da hora prevista em UTC (ISO-8601)")
    forecast_for_brt: str = Field(
        description="Início da hora prevista em Brasília (ISO-8601)"
    )
    forecast_close_utc: str = Field(
        description="Fechamento da hora prevista em UTC (ISO-8601)"
    )
    forecast_close_brt: str = Field(
        description="Fechamento da hora prevista em Brasília (ISO-8601)"
    )
    confidence_interval_95_usd: ConfidenceIntervalResponse | None = Field(
        default=None,
        description="Intervalo de confiança estimado de 95%",
    )
    estimated_error_pct: float | None = Field(
        default=None,
        description="Erro percentual estimado com base nas métricas do modelo",
    )
    data_source: str = Field(
        description="Fonte dos dados de mercado usada: yfinance ou binance"
    )
    processing_time_ms: float = Field(
        description="Tempo de processamento da requisição em milissegundos"
    )


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str = Field(
        description=(
            "healthy quando todos os checks passam; "
            "caso contrário degraded"
        )
    )
    artifacts_ready: bool = Field(description="Modelo e scaler carregados")
    model_usable: bool = Field(description="Modelo responde a uma inferência de sanidade")
    market_data_accessible: bool = Field(description="Consulta de mercado disponível")
    data_source: str | None = Field(
        default=None,
        description="Fonte de dados ativa: yfinance ou binance",
    )
    last_market_timestamp_utc: str | None = Field(
        default=None,
        description="Último candle válido em UTC (ISO-8601)",
    )
    last_market_timestamp_brt: str | None = Field(
        default=None,
        description="Último candle válido em Brasília (ISO-8601)",
    )
    cpu_usage: float = Field(description="Uso atual de CPU (%)")
    memory_usage: float = Field(description="Uso atual de memória (%)")
    details: str | None = Field(
        default=None,
        description="Detalhes quando status=degraded",
    )


class LiveResponse(BaseModel):
    status: str = Field(description="alive quando a API está respondendo")
    artifacts_ready: bool = Field(description="Modelo e scaler carregados")


class PredictionLogEntry(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    requested_at_utc: str
    ticker: str
    input_mode: str
    last_input_candle_utc: str
    forecast_for_utc: str
    predicted_price_usd: float
    data_source: str
    processing_time_ms: float


class PredictionHistoryResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    total_logged: int = Field(description="Total de previsões armazenadas no histórico")
    predictions: list[PredictionLogEntry] = Field(
        description=(
            "Últimas previsões realizadas "
            "(mais recente primeiro)"
        )
    )


class ChatRequest(BaseModel):
    message: str = Field(
        description="Pergunta ou instrução em linguagem natural para o agente LLM"
    )


class AgentStepResponse(BaseModel):
    tool: str = Field(description="Nome da ferramenta executada")
    tool_input: str = Field(description="Entrada fornecida à ferramenta")
    observation: str = Field(description="Resultado retornado pela ferramenta")


class ChatResponse(BaseModel):
    response: str = Field(description="Resposta final do agente em linguagem natural")
    steps: list[AgentStepResponse] = Field(
        default_factory=list,
        description="Passos intermediários executados pelo agente (ferramentas chamadas)",
    )


class DriftCheckRequest(BaseModel):
    ticker: str = Field(default="BTC-USD", description="Ticker para verificacao de drift")


# --- Configuração geral ---
ml_artifacts: dict[str, Any] = {}
SUPPORTED_TICKER = "BTC-USD"
LOOKBACK = 60
MODEL_PATH = "models/lstm_btc_hourly.keras"
SCALER_PATH = "models/scaler_btc.gz"
SCALER_RETURN_PATH = "models/scaler_btc_return.gz"
MODEL_META_PATH = "models/model_metadata_btc.json"
CACHE_TTL_SECONDS = 30
YFINANCE_TIMEOUT_SECONDS = 10
YFINANCE_MAX_RETRIES = 3
BINANCE_SYMBOL = "BTCUSDT"
BINANCE_API_URL = "https://api.binance.com/api/v3/klines"
BINANCE_TIMEOUT_SECONDS = 10
PREDICTIONS_HISTORY_MAX = 100
market_cache: dict[str, dict[str, Any]] = {}
BRASILIA_TZ = "America/Sao_Paulo"

# Histórico circular de predições (MLOps)
prediction_log: collections.deque[dict[str, Any]] = collections.deque(
    maxlen=PREDICTIONS_HISTORY_MAX
)

# --- Prometheus Metrics ---
_prom_registry = CollectorRegistry()
METRIC_PREDICT_REQUESTS = Counter(
    "stockcast_predict_requests_total",
    "Total de chamadas ao endpoint /predict",
    ["ticker", "status"],
    registry=_prom_registry,
)
METRIC_PREDICT_LATENCY = Histogram(
    "stockcast_predict_latency_seconds",
    "Latência de inferência do endpoint /predict em segundos",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    registry=_prom_registry,
)
METRIC_DATA_SOURCE = Counter(
    "stockcast_market_data_source_total",
    "Contagem de requisições por fonte de dados de mercado",
    ["source"],
    registry=_prom_registry,
)
METRIC_DATA_ERRORS = Counter(
    "stockcast_market_data_errors_total",
    "Total de falhas ao buscar dados de mercado",
    ["source"],
    registry=_prom_registry,
)
METRIC_CPU = Gauge(
    "stockcast_cpu_usage_percent", "Uso de CPU do processo (%)", registry=_prom_registry
)
METRIC_MEMORY = Gauge(
    "stockcast_memory_usage_percent", "Uso de memória do processo (%)", registry=_prom_registry
)


# --- Utilitários ---
def remove_incomplete_hour_candle(series: pd.Series) -> pd.Series:
    if len(series) < 2:
        return series

    last_ts = pd.Timestamp(series.index[-1])
    now_utc = pd.Timestamp.utcnow()

    if last_ts.tzinfo is None:
        now_ref = now_utc.tz_localize(None)
    else:
        now_ref = now_utc.tz_convert(last_ts.tz)

    if last_ts >= now_ref.floor("h"):
        return series.iloc[:-1]
    return series


def get_cached_market_data(ticker: str) -> pd.DataFrame | None:
    cache_entry = market_cache.get(ticker)
    if not cache_entry:
        return None

    age_seconds = time.time() - cache_entry["cached_at"]
    if age_seconds > CACHE_TTL_SECONDS:
        return None

    return cache_entry["data"].copy()


def set_cached_market_data(ticker: str, data: pd.DataFrame, source: str) -> None:
    market_cache[ticker] = {"cached_at": time.time(), "data": data.copy(), "source": source}


def get_cached_source(ticker: str) -> str:
    entry = market_cache.get(ticker)
    if not entry:
        return "unknown"
    return str(entry.get("source", "unknown"))


def timestamp_to_utc_iso(ts: pd.Timestamp) -> str:
    ts = pd.Timestamp(ts)
    ts_utc = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return str(ts_utc.isoformat())


def timestamp_to_brt_iso(ts: pd.Timestamp) -> str:
    ts = pd.Timestamp(ts)
    ts_utc = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return str(ts_utc.tz_convert(ZoneInfo(BRASILIA_TZ)).isoformat())


def estimate_uncertainty(
    predicted_price: float, metadata: dict[str, Any]
) -> tuple[float | None, ConfidenceIntervalResponse | None]:
    metrics = metadata.get("metrics", {}) if isinstance(metadata, dict) else {}

    mape_price = metrics.get("mape_price")
    rmse_price = metrics.get("rmse_price")

    estimated_error_pct = None
    if mape_price is not None:
        estimated_error_pct = float(mape_price)
    elif rmse_price is not None and predicted_price > 0:
        estimated_error_pct = float((float(rmse_price) / predicted_price) * 100)

    if rmse_price is not None:
        margin = 1.96 * float(rmse_price)
    elif estimated_error_pct is not None:
        margin = predicted_price * (estimated_error_pct / 100)
    else:
        return estimated_error_pct, None

    ci = ConfidenceIntervalResponse(
        low_usd=round(max(0.0, predicted_price - margin), 2),
        high_usd=round(predicted_price + margin, 2),
    )
    return estimated_error_pct, ci


def is_production_environment() -> bool:
    for env_var in PRODUCTION_ENV_VARIABLES:
        value = os.getenv(env_var)
        if value and value.strip().lower() in PRODUCTION_ENV_NAMES:
            return True
    return False


def validate_google_api_key_for_startup() -> None:
    if not is_production_environment():
        return

    api_key = (os.getenv("GOOGLE_API_KEY") or "").strip()
    if api_key.lower() in INSECURE_GOOGLE_API_KEY_VALUES:
        raise RuntimeError(
            "GOOGLE_API_KEY inválida para produção. Configure uma chave real e segura."
        )

    if not GOOGLE_API_KEY_PATTERN.fullmatch(api_key):
        raise RuntimeError(
            "GOOGLE_API_KEY com formato inválido para produção. "
            "Use uma chave válida do provedor antes de iniciar a API."
        )


def load_trained_model(model_path: str) -> Any:
    keras_module = getattr(tf, "keras", None)
    if keras_module is None or not hasattr(keras_module, "models"):
        raise RuntimeError("TensorFlow/Keras indisponível para carregar o modelo")
    return keras_module.models.load_model(model_path)


# --- Fontes de dados ---
def _download_from_yfinance(ticker: str) -> pd.DataFrame:
    """Baixa dados do Yahoo Finance. Lança exceção em caso de falha."""
    df = yf.download(
        ticker, period="1mo", interval="1h", progress=False, timeout=YFINANCE_TIMEOUT_SECONDS
    )
    if df is None or df.empty:
        raise ValueError("Resposta vazia do Yahoo Finance")

    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs(ticker, axis=1, level=1)
        except KeyError:
            df.columns = df.columns.get_level_values(0)

    if isinstance(df, pd.Series):
        df = df.to_frame(name="Close")

    return df


def _download_from_binance(limit: int = 200) -> pd.DataFrame:
    """Baixa candles horários do BTCUSDT via Binance REST API pública (sem autenticação).
    Retorna DataFrame com índice DatetimeIndex UTC e colunas Close, High, Low, Volume.
    """
    resp = requests.get(
        BINANCE_API_URL,
        params={"symbol": BINANCE_SYMBOL, "interval": "1h", "limit": limit},
        timeout=BINANCE_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    raw = resp.json()
    if not raw:
        raise ValueError("Resposta vazia da Binance")

    # Cada item: [open_time, open, high, low, close, volume, ...]
    timestamps = pd.to_datetime([row[0] for row in raw], unit="ms", utc=True)
    df = pd.DataFrame(
        {
            "Close": [float(row[4]) for row in raw],
            "High": [float(row[2]) for row in raw],
            "Low": [float(row[3]) for row in raw],
            "Volume": [float(row[5]) for row in raw],
        },
        index=timestamps,
    )
    df.index.name = "Datetime"
    return df


def download_with_retry(ticker: str) -> tuple[pd.DataFrame, str]:
    """Retorna (DataFrame, source) com fallback automático para a Binance."""
    cached = get_cached_market_data(ticker)
    if cached is not None:
        return cached, get_cached_source(ticker)

    # Tentativa primária: Yahoo Finance
    last_error = None
    for attempt in range(1, YFINANCE_MAX_RETRIES + 1):
        try:
            df = _download_from_yfinance(ticker)
            set_cached_market_data(ticker, df, "yfinance")
            METRIC_DATA_SOURCE.labels(source="yfinance").inc()
            logger.info("Dados de mercado obtidos via Yahoo Finance (tentativa %d)", attempt)
            return df, "yfinance"
        except Exception as error:
            last_error = error
            logger.warning(
                "Yahoo Finance falhou (tentativa %d/%d): %s", attempt, YFINANCE_MAX_RETRIES, error
            )
            METRIC_DATA_ERRORS.labels(source="yfinance").inc()
            if attempt < YFINANCE_MAX_RETRIES:
                time.sleep(0.5 * attempt)

    # Fallback: Binance
    logger.warning(
        "Yahoo Finance indisponível após %d tentativas. Tentando Binance...",
        YFINANCE_MAX_RETRIES,
    )
    try:
        df_binance = _download_from_binance()
        set_cached_market_data(ticker, df_binance, "binance")
        METRIC_DATA_SOURCE.labels(source="binance").inc()
        logger.info("Dados de mercado obtidos via Binance (fallback)")
        return df_binance, "binance"
    except Exception as binance_error:
        logger.error("Binance fallback também falhou: %s", binance_error)
        METRIC_DATA_ERRORS.labels(source="binance").inc()

    raise HTTPException(
        status_code=503,
        detail=(
            "Falha ao consultar dados de mercado em todas as fontes disponíveis "
            "(Yahoo Finance e Binance)"
        ),
    ) from last_error


# --- Health checks ---
def perform_health_checks() -> dict[str, Any]:
    model: Any | None = ml_artifacts.get("model")
    scaler = ml_artifacts.get("scaler")

    artifacts_ready = model is not None and scaler is not None
    model_usable = False
    market_data_accessible = False
    active_source = None
    last_market_timestamp_utc = None
    last_market_timestamp_brt = None
    issues = []

    if artifacts_ready:
        try:
            metadata_check = ml_artifacts.get("metadata", {})
            n_features = metadata_check.get("n_features", 1)
            sample_input = np.zeros((1, LOOKBACK, n_features), dtype=np.float32)
            if model is None:
                raise ValueError("Modelo indisponível")
            prediction = model.predict(sample_input, verbose=0)
            if prediction is None or len(prediction) == 0:
                raise ValueError("Predição vazia do modelo")
            model_usable = True
        except Exception:
            issues.append("Modelo carregado, mas não respondeu a inferência de saúde")
    else:
        issues.append("Artefatos de modelo/scaler não carregados")

    try:
        df, active_source = download_with_retry(SUPPORTED_TICKER)
        close_series = df["Close"].dropna()
        close_series = remove_incomplete_hour_candle(close_series)

        if len(close_series) == 0:
            raise ValueError("Sem candles válidos")

        market_data_accessible = True
        last_market_ts = pd.Timestamp(close_series.index[-1])
        last_market_timestamp_utc = timestamp_to_utc_iso(last_market_ts)
        last_market_timestamp_brt = timestamp_to_brt_iso(last_market_ts)
    except HTTPException:
        issues.append("Dados de mercado indisponíveis em todas as fontes")
    except Exception:
        issues.append("Dados de mercado indisponíveis no momento")

    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory().percent
    METRIC_CPU.set(cpu)
    METRIC_MEMORY.set(mem)

    healthy = artifacts_ready and model_usable and market_data_accessible
    return {
        "status": "healthy" if healthy else "degraded",
        "artifacts_ready": artifacts_ready,
        "model_usable": model_usable,
        "market_data_accessible": market_data_accessible,
        "data_source": active_source,
        "last_market_timestamp_utc": last_market_timestamp_utc,
        "last_market_timestamp_brt": last_market_timestamp_brt,
        "cpu_usage": cpu,
        "memory_usage": mem,
        "details": None if healthy else " | ".join(issues),
    }


# --- Lifecycle ---
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    validate_google_api_key_for_startup()
    logger.info("Carregando modelo LSTM Hourly e scaler...")
    try:
        ml_artifacts["model"] = load_trained_model(MODEL_PATH)
        ml_artifacts["scaler"] = joblib.load(SCALER_PATH)

        # Scaler separado de log_return para inversão da predição (modelos multi-feature)
        if os.path.exists(SCALER_RETURN_PATH):
            ml_artifacts["scaler_return"] = joblib.load(SCALER_RETURN_PATH)
        else:
            ml_artifacts["scaler_return"] = None

        try:
            with open(MODEL_META_PATH, encoding="utf-8") as meta_file:
                ml_artifacts["metadata"] = json.load(meta_file)
        except FileNotFoundError:
            ml_artifacts["metadata"] = {
                "target": "log_return",
                "lookback": LOOKBACK,
                "ticker": SUPPORTED_TICKER,
            }

        logger.info("Artefatos carregados com sucesso.")
    except Exception as e:
        ml_artifacts.clear()
        raise RuntimeError(f"Falha crítica ao carregar artefatos do modelo: {e}") from e
    yield
    ml_artifacts.clear()
    logger.info("Artefatos descarregados. API encerrada.")


app = FastAPI(title="Bitcoin Hourly Forecaster", version="3.0.0", lifespan=lifespan)


# --- Endpoints ---
@app.get(
    "/live",
    response_model=LiveResponse,
    summary="Liveness da API",
    description="Endpoint leve para healthcheck de container, sem consulta externa.",
)
def live_check() -> dict[str, Any]:
    artifacts_ready = "model" in ml_artifacts and "scaler" in ml_artifacts
    return {"status": "alive", "artifacts_ready": artifacts_ready}


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Saúde efetiva da API",
    description=(
        "Valida artefatos, inferência do modelo e acesso ao mercado. "
        "Indica a fonte de dados ativa (yfinance ou binance). "
        "Retorna timestamps em UTC e Brasília."
    ),
)
def health_check() -> dict[str, Any]:
    return perform_health_checks()


@app.get(
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
        content=generate_latest(_prom_registry).decode("utf-8"), media_type=CONTENT_TYPE_LATEST
    )


@app.get(
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


@app.post(
    "/admin/check-drift",
    summary="Executa checagem de data drift",
    description=(
        "Executa a deteccao de data drift via PSI comparando historico de previsoes e dados reais. "
        "Endpoint administrativo para automacao MLOps."
    ),
)
async def check_drift(
    request: DriftCheckRequest = Body(default_factory=DriftCheckRequest),  # noqa: B008
) -> dict[str, Any]:
    """Executa a checagem assíncrona de drift.

    Este endpoint deve ser chamado por um EventBridge/Cronjob na AWS para automacao do MLOps.
    """
    result = await detect_data_drift(
        ticker=request.ticker.upper(),
        download_fn=download_with_retry,
        prediction_log=prediction_log,
    )

    automation_summary = process_drift_result(
        result,
        DriftAutomationConfig.from_sources(),
        mlflow_module=mlflow,
    )

    return {**result, "automation": automation_summary}


@app.post(
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
    ),  # noqa: B008
) -> dict[str, Any]:
    start_proc = time.perf_counter()
    ticker = request.ticker.upper()

    if ticker != SUPPORTED_TICKER:
        METRIC_PREDICT_REQUESTS.labels(ticker=ticker, status="error_unsupported").inc()
        raise HTTPException(
            status_code=400, detail=f"Este modelo foi treinado apenas para {SUPPORTED_TICKER}."
        )

    if "model" not in ml_artifacts or "scaler" not in ml_artifacts:
        METRIC_PREDICT_REQUESTS.labels(ticker=ticker, status="error_no_model").inc()
        raise HTTPException(status_code=503, detail="Modelo não disponível.")

    try:
        # 1. Coleta de dados de mercado com fallback automático
        df, data_source = download_with_retry(ticker)

        if "Close" not in df.columns:
            raise HTTPException(status_code=503, detail="Dados de mercado sem coluna Close")

        metadata = ml_artifacts.get("metadata", {})
        n_features = metadata.get("n_features", 1)
        scaler = ml_artifacts["scaler"]
        scaler_return = ml_artifacts.get("scaler_return")
        model = ml_artifacts["model"]

        close_series = df["Close"].dropna()
        if not request.use_partial_candle:
            close_series = remove_incomplete_hour_candle(close_series)

        required_points = LOOKBACK + 1
        if len(close_series) < required_points:
            raise HTTPException(
                status_code=400,
                detail=f"Dados insuficientes para janela de retorno ({required_points} closes).",
            )

        if n_features > 1:
            # Inferência multi-feature: reconstruir indicadores técnicos
            if not {"High", "Low", "Volume"}.issubset(df.columns):
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Dados de mercado sem colunas OHLCV necessárias "
                        "para inferência multi-feature."
                    ),
                )
            ohlcv = df[["Close", "High", "Low", "Volume"]].dropna()
            if not request.use_partial_candle:
                ohlcv = ohlcv.loc[close_series.index]

            # Reuso do cálculo compartilhado de features técnicas.
            features_df = _build_feature_matrix(ohlcv)

            if len(features_df) < LOOKBACK:
                raise HTTPException(
                    status_code=400,
                    detail=f"Dados insuficientes para janela multi-feature de {LOOKBACK}h.",
                )

            window = features_df.to_numpy()[-LOOKBACK:]
            scaled_input = scaler.transform(window)
            X_input = scaled_input.reshape(1, LOOKBACK, n_features)

            predicted_scaled = model.predict(X_input, verbose=0)
            # Inverter apenas a coluna log_return (índice 0) usando scaler_return
            if scaler_return is not None:
                predicted_log_return = float(
                    scaler_return.inverse_transform(predicted_scaled.reshape(-1, 1)).reshape(-1)[0]
                )
            else:
                # Fallback: inverter via scaler_all usando feature 0 (log_return)
                try:
                    min_val = float(scaler.data_min_[0])
                    max_val = float(scaler.data_max_[0])
                    predicted_log_return = (
                        float(predicted_scaled.reshape(-1)[0]) * (max_val - min_val) + min_val
                    )
                except (AttributeError, IndexError):
                    predicted_log_return = float(predicted_scaled.reshape(-1)[0])

            last_close = float(ohlcv["Close"].iloc[-1])
            last_observed_ts = pd.Timestamp(features_df.index[-1])

        else:
            # Inferência single-feature (modelo legado com apenas log_return)
            log_price_series = pd.Series(np.log(close_series.values), index=close_series.index)
            return_series = log_price_series.diff().dropna()

            if len(return_series) < LOOKBACK:
                raise HTTPException(
                    status_code=400,
                    detail=f"Dados insuficientes para gerar janela de retorno de {LOOKBACK}h.",
                )

            last_returns = np.asarray(
                return_series.to_numpy()[-LOOKBACK:],
                dtype=float,
            ).reshape(-1, 1)
            scaled_input = scaler.transform(last_returns)
            X_input = scaled_input.reshape(1, LOOKBACK, 1)

            predicted_scaled = model.predict(X_input, verbose=0)
            predicted_log_return = float(scaler.inverse_transform(predicted_scaled).reshape(-1)[0])
            last_close = float(close_series.iloc[-1])
            last_observed_ts = pd.Timestamp(close_series.index[-1])

        # 3. Conversão para preço e metadados temporais
        forecast_for_ts = last_observed_ts + pd.Timedelta(hours=1)
        forecast_close_ts = forecast_for_ts + pd.Timedelta(hours=1) - pd.Timedelta(seconds=1)
        predicted_price = last_close * np.exp(predicted_log_return)

        estimated_error_pct, confidence_interval_95 = estimate_uncertainty(
            float(predicted_price),
            metadata,
        )

        proc_time = (time.perf_counter() - start_proc) * 1000
        METRIC_PREDICT_LATENCY.observe(proc_time / 1000)
        METRIC_PREDICT_REQUESTS.labels(ticker=ticker, status="success").inc()

        # 4. Log de predição para histórico/auditoria (MLOps)
        prediction_log.append(
            {
                "requested_at_utc": pd.Timestamp.utcnow().isoformat(),
                "ticker": ticker,
                "input_mode": (
                    "include_partial_candle"
                    if request.use_partial_candle
                    else "closed_candles_only"
                ),
                "last_input_candle_utc": timestamp_to_utc_iso(last_observed_ts),
                "forecast_for_utc": timestamp_to_utc_iso(forecast_for_ts),
                "predicted_price_usd": round(float(predicted_price), 2),
                "data_source": data_source,
                "processing_time_ms": round(proc_time, 2),
            }
        )
        logger.info(
            "Previsão gerada | ticker=%s source=%s price=%.2f latency=%.1fms",
            ticker,
            data_source,
            float(predicted_price),
            proc_time,
        )

        return {
            "ticker": ticker,
            "prediction_type": "Next Hour Close",
            "input_mode": (
                "include_partial_candle" if request.use_partial_candle else "closed_candles_only"
            ),
            "last_input_candle_utc": timestamp_to_utc_iso(last_observed_ts),
            "last_input_candle_brt": timestamp_to_brt_iso(last_observed_ts),
            "predicted_price_usd": round(float(predicted_price), 2),
            "forecast_for_utc": timestamp_to_utc_iso(forecast_for_ts),
            "forecast_for_brt": timestamp_to_brt_iso(forecast_for_ts),
            "forecast_close_utc": timestamp_to_utc_iso(forecast_close_ts),
            "forecast_close_brt": timestamp_to_brt_iso(forecast_close_ts),
            "confidence_interval_95_usd": confidence_interval_95,
            "estimated_error_pct": (
                None if estimated_error_pct is None else round(float(estimated_error_pct), 2)
            ),
            "data_source": data_source,
            "processing_time_ms": round(proc_time, 2),
        }

    except HTTPException:
        METRIC_PREDICT_REQUESTS.labels(ticker=ticker, status="error").inc()
        raise
    except Exception as e:
        METRIC_PREDICT_REQUESTS.labels(ticker=ticker, status="error_internal").inc()
        logger.error("Erro interno em /predict: %s: %s", type(e).__name__, e)
        raise HTTPException(status_code=500, detail="Falha interna ao gerar previsão") from None


@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="Chat com o Agente LLM (ReAct)",
    description=(
        "Recebe uma mensagem em linguagem natural e a processa com um Agente ReAct "
        "(LangChain + Gemini). O agente orquestra 3 ferramentas: PrevisaoBitcoinTool "
        "(inferência LSTM), CotacaoAtualTool (cotação em tempo real) e CryptoRAGTool "
        "(contexto de mercado simulado). Requer a variável de ambiente GOOGLE_API_KEY."
    ),
)
def chat(request: ChatRequest) -> dict[str, Any]:
    """Endpoint de chat com o Agente ReAct LangChain."""
    input_validation = input_guardrail.validate(request.message)
    if not input_validation.allowed:
        raise HTTPException(status_code=400, detail=input_validation.reason)

    # Import lazy para evitar importação circular no nível de módulo
    from src.agent.react_agent import build_agent  # noqa: PLC0415

    try:
        agent_executor = build_agent(ml_artifacts)
    except OSError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        result = agent_executor.invoke({"input": request.message})
    except Exception as exc:
        logger.error("Erro no agente ReAct: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro no agente: {exc}") from exc

    # Extrair passos intermediários
    steps: list[dict[str, str]] = []
    for action, observation in result.get("intermediate_steps", []):
        steps.append(
            {
                "tool": getattr(action, "tool", str(action)),
                "tool_input": str(getattr(action, "tool_input", "")),
                "observation": output_guardrail.sanitize(str(observation)),
            }
        )

    safe_output = output_guardrail.sanitize(result.get("output", ""))

    return {"response": safe_output, "steps": steps}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
