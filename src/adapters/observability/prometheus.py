from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

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
