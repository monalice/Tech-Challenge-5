"""Fontes de dados de mercado com Strategy Pattern.

Implementa o padrão Strategy para aquisição de dados OHLCV de criptoativos:

- :class:`MarketDataSource` — Protocolo de contrato para todas as fontes.
- :class:`YFinanceSource`   — Implementação via Yahoo Finance.
- :class:`BinanceSource`    — Implementação via Binance REST API pública.
- :class:`FallbackMarketData` — Agregador com retry e fallback automático.

Uso típico::

    from src.infrastructure.market_data import (
        BinanceSource,
        FallbackMarketData,
        YFinanceSource,
    )

    fetcher = FallbackMarketData(
        primary=YFinanceSource(),
        fallback=BinanceSource(),
        max_retries=3,
    )
    df, source = fetcher.download("BTC-USD")
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Protocol, runtime_checkable

import pandas as pd
import requests
import yfinance as yf

from src.domain.constants import (
    BINANCE_API_URL,
    BINANCE_SYMBOL,
    BINANCE_TIMEOUT_SECONDS,
    YFINANCE_TIMEOUT_SECONDS,
)

logger = logging.getLogger("stockcast.market_data")


# ---------------------------------------------------------------------------
# Protocolo (interface Strategy)
# ---------------------------------------------------------------------------


@runtime_checkable
class MarketDataSource(Protocol):
    """Protocolo de contrato para fontes de dados de mercado intercambiáveis.

    Qualquer classe que implemente :attr:`name` e :meth:`fetch` satisfaz este
    protocolo e pode ser usada como estratégia em :class:`FallbackMarketData`.
    """

    #: Identificador legível da fonte (ex: ``"yfinance"``, ``"binance"``).
    name: str

    def fetch(self, ticker: str) -> pd.DataFrame:
        """Baixa dados OHLCV horários para o ticker informado.

        Args:
            ticker: Símbolo do ativo (ex: ``"BTC-USD"``).

        Returns:
            DataFrame com pelo menos a coluna ``Close`` e índice DatetimeIndex.

        Raises:
            ValueError: Quando a resposta da fonte está vazia.
            Exception: Em caso de falha de rede ou API.
        """
        ...


# ---------------------------------------------------------------------------
# Implementações concretas
# ---------------------------------------------------------------------------


class YFinanceSource:
    """Fonte de dados via Yahoo Finance (yfinance).

    Baixa candles horários do último mês para o ticker informado.
    Lida automaticamente com respostas MultiIndex retornadas por versões
    recentes do yfinance quando múltiplos tickers são consultados.
    """

    name: str = "yfinance"

    def fetch(self, ticker: str) -> pd.DataFrame:
        """Baixa dados horários do Yahoo Finance para o último mês.

        Args:
            ticker: Símbolo do ativo (ex: ``"BTC-USD"``).

        Returns:
            DataFrame com pelo menos a coluna ``Close`` indexado por DatetimeIndex.

        Raises:
            ValueError: Se a resposta do Yahoo Finance estiver vazia.
        """
        df = yf.download(
            ticker,
            period="1mo",
            interval="1h",
            progress=False,
            timeout=YFINANCE_TIMEOUT_SECONDS,
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


class BinanceSource:
    """Fonte de dados via Binance REST API pública (sem autenticação).

    Baixa candles horários do par BTCUSDT via endpoint ``/api/v3/klines``.
    O parâmetro ``ticker`` de :meth:`fetch` é aceito por conformidade com o
    Protocolo, mas ignorado — a fonte sempre consulta ``BINANCE_SYMBOL``.
    """

    name: str = "binance"

    def __init__(self, limit: int = 200) -> None:
        """Configura o número de candles a retornar por requisição.

        Args:
            limit: Número de candles (máximo 1000 pela API da Binance).
        """
        self._limit = limit

    def fetch(self, ticker: str) -> pd.DataFrame:  # noqa: ARG002  (ticker ignorado)
        """Baixa candles horários do BTCUSDT via Binance REST API.

        Args:
            ticker: Ignorado — a fonte sempre consulta ``BTCUSDT``.

        Returns:
            DataFrame com índice DatetimeIndex UTC e colunas
            ``Close``, ``High``, ``Low``, ``Volume``.

        Raises:
            ValueError: Se a resposta da Binance estiver vazia.
            requests.HTTPError: Se a requisição HTTP falhar.
        """
        resp = requests.get(
            BINANCE_API_URL,
            params={"symbol": BINANCE_SYMBOL, "interval": "1h", "limit": self._limit},
            timeout=BINANCE_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        raw = resp.json()
        if not raw:
            raise ValueError("Resposta vazia da Binance")

        # Formato Binance kline: [open_time, open, high, low, close, volume, ...]
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


# ---------------------------------------------------------------------------
# Agregador com retry + fallback
# ---------------------------------------------------------------------------


class FallbackMarketData:
    """Gerencia tentativas (retries) e fallback entre fontes de dados de mercado.

    Implementa o Strategy Pattern com duas estratégias configuráveis:
    uma primária (tentada ``max_retries`` vezes com back-off exponencial) e
    uma de fallback (tentada uma vez em caso de falha total da primária).

    Em caso de falha em ambas as fontes, lança :class:`RuntimeError` —
    a camada de entrega (FastAPI) é responsável por traduzir para HTTP 503.

    Args:
        primary: Fonte primária de dados (ex: :class:`YFinanceSource`).
        fallback: Fonte de fallback (ex: :class:`BinanceSource`).
        max_retries: Número máximo de tentativas na fonte primária.
        sleep_fn: Função de espera entre tentativas. Substituível em testes
            para eliminar sleeps reais (``sleep_fn=lambda _: None``).
    """

    def __init__(
        self,
        primary: MarketDataSource,
        fallback: MarketDataSource,
        max_retries: int = 3,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._max_retries = max_retries
        self._sleep = sleep_fn

    def download(self, ticker: str) -> tuple[pd.DataFrame, str]:
        """Baixa dados de mercado com retry na primária e fallback automático.

        Tenta a fonte primária até :attr:`max_retries` vezes, com back-off
        linear (``0.5 * attempt`` segundos entre tentativas). Em caso de falha
        persistente, tenta a fonte de fallback uma vez.

        Args:
            ticker: Símbolo do ativo (ex: ``"BTC-USD"``).

        Returns:
            Tupla ``(DataFrame, source_name)`` onde *source_name* é o
            :attr:`~MarketDataSource.name` da fonte que respondeu com sucesso.

        Raises:
            RuntimeError: Se todas as fontes falharem. A mensagem inclui os
                erros de ambas as fontes para facilitar o diagnóstico.
        """
        last_err: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                df = self._primary.fetch(ticker)
                logger.info(
                    "[market] %s: dados obtidos (tentativa %d/%d)",
                    self._primary.name,
                    attempt,
                    self._max_retries,
                )
                return df, self._primary.name
            except Exception as exc:
                last_err = exc
                logger.warning(
                    "[market] %s tentativa %d/%d falhou: %s",
                    self._primary.name,
                    attempt,
                    self._max_retries,
                    exc,
                )
                if attempt < self._max_retries:
                    self._sleep(0.5 * attempt)

        logger.warning(
            "[market] %s indisponível após %d tentativas. Fallback para %s...",
            self._primary.name,
            self._max_retries,
            self._fallback.name,
        )

        try:
            df = self._fallback.fetch(ticker)
            logger.info("[market] %s: dados obtidos (fallback)", self._fallback.name)
            return df, self._fallback.name
        except Exception as exc:
            raise RuntimeError(
                f"Todas as fontes de dados falharam. "
                f"Último erro {self._primary.name}: {last_err}. "
                f"{self._fallback.name}: {exc}"
            ) from exc
