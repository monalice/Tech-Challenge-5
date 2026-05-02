from pydantic import BaseModel, ConfigDict, Field


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
