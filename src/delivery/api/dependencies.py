import collections
import logging
from typing import Any

import pandas as pd
from fastapi import HTTPException

from src.adapters.observability.prometheus import (
    METRIC_DATA_ERRORS,
    METRIC_DATA_SOURCE,
)
from src.delivery.api.schemas import ConfidenceIntervalResponse
from src.domain.inference import estimate_uncertainty as _estimate_uncertainty_domain
from src.infrastructure.market_data import BinanceSource, FallbackMarketData, YFinanceSource
from src.use_cases.market_cache import (
    get_cached_market_data,
    get_cached_source,
    set_cached_market_data,
)

logger = logging.getLogger("stockcast")

YFINANCE_MAX_RETRIES = 3
PREDICTIONS_HISTORY_MAX = 100

# Instância compartilhada da estratégia de mercado (primária + fallback)
_APP_MARKET_DATA: FallbackMarketData = FallbackMarketData(
    primary=YFinanceSource(),
    fallback=BinanceSource(),
    max_retries=YFINANCE_MAX_RETRIES,
)

# Histórico circular de predições (MLOps)
prediction_log: collections.deque[dict[str, Any]] = collections.deque(
    maxlen=PREDICTIONS_HISTORY_MAX
)


def download_with_retry(ticker: str) -> tuple[pd.DataFrame, str]:
    """Baixa dados de mercado com retry no Yahoo Finance e fallback para Binance.

    Verifica o cache em memória antes de realizar qualquer requisição de rede.
    """
    cached = get_cached_market_data(ticker)
    if cached is not None:
        return cached, get_cached_source(ticker)

    try:
        df, source = _APP_MARKET_DATA.download(ticker)
    except RuntimeError as exc:
        logger.error("Todas as fontes de dados falharam: %s", exc)
        METRIC_DATA_ERRORS.labels(source="yfinance").inc()
        METRIC_DATA_ERRORS.labels(source="binance").inc()
        raise HTTPException(
            status_code=503,
            detail=(
                "Falha ao consultar dados de mercado em todas as fontes disponíveis "
                "(Yahoo Finance e Binance)"
            ),
        ) from exc

    set_cached_market_data(ticker, df, source)
    METRIC_DATA_SOURCE.labels(source=source).inc()
    return df, source


class _DownloadWithRetryPort:
    """Adaptador que expõe :func:`download_with_retry` como MarketDataPort.

    Usa late binding — resolve ``download_with_retry`` no namespace deste módulo
    no momento da chamada, garantindo que monkeypatches em testes sejam aplicados
    corretamente.
    """

    def download(self, ticker: str) -> tuple[pd.DataFrame, str]:
        import src.delivery.api.dependencies as _self  # noqa: PLC0415

        return _self.download_with_retry(ticker)


def estimate_uncertainty(
    predicted_price: float, metadata: dict[str, Any]
) -> tuple[float | None, ConfidenceIntervalResponse | None]:
    """Wrapper para compatibilidade retroativa com testes existentes."""
    err_pct, ci = _estimate_uncertainty_domain(predicted_price, metadata)
    if ci is None:
        return err_pct, None
    return err_pct, ConfidenceIntervalResponse(low_usd=ci.low_usd, high_usd=ci.high_usd)
