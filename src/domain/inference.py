"""Serviço de inferência do modelo LSTM — lógica pura de ML, sem dependências HTTP.

Contém:
    - :class:`ConfidenceInterval` — valor de domínio para o IC de 95%.
    - :class:`InferenceResult`  — resultado completo de uma inferência.
    - :class:`DataServiceError` — dados de mercado incompletos (origem: fonte externa).
    - :class:`InsufficientDataError` — janela temporal insuficiente (origem: parâmetros).
    - :func:`estimate_uncertainty` — cálculo puro de MAPE/RMSE → IC.
    - :class:`InferenceService` — orquestra coleta → features → modelo → inversão → IC.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.domain.constants import LOOKBACK, Z_SCORE_95_CONFIDENCE
from src.domain.ports import LoadedArtifacts, MarketDataPort
from src.domain.time_utils import remove_incomplete_hour_candle
from src.features.technical_features import build_feature_matrix

logger = logging.getLogger("stockcast.inference")


# ---------------------------------------------------------------------------
# Exceções de domínio
# ---------------------------------------------------------------------------


class DataServiceError(RuntimeError):
    """Dados de mercado recebidos sem as colunas obrigatórias (HTTP 503 na camada web)."""


class InsufficientDataError(ValueError):
    """Janela temporal insuficiente para gerar a sequência de entrada (HTTP 400 na camada web)."""


# ---------------------------------------------------------------------------
# Tipos de valor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfidenceInterval:
    """Intervalo de confiança de 95% para a previsão de preço."""

    low_usd: float
    high_usd: float


@dataclass(frozen=True)
class InferenceResult:
    """Resultado completo de uma inferência do modelo LSTM."""

    predicted_price_usd: float
    last_close: float
    last_observed_ts: pd.Timestamp
    data_source: str
    estimated_error_pct: float | None
    confidence_interval: ConfidenceInterval | None


# ---------------------------------------------------------------------------
# Lógica pura de incerteza
# ---------------------------------------------------------------------------


def estimate_uncertainty(
    predicted_price: float,
    metadata: dict[str, Any],
) -> tuple[float | None, ConfidenceInterval | None]:
    """Estima o erro percentual e o IC de 95% a partir dos metadados do modelo.

    Usa MAPE quando disponível; caso contrário, deriva erro percentual via RMSE.
    O intervalo de confiança é ``predicted_price ± Z_95 * rmse``.

    Args:
        predicted_price: Preço previsto em USD.
        metadata: Dicionário de metadados com chave ``"metrics"`` contendo
            opcionalmente ``"mape_price"`` e ``"rmse_price"``.

    Returns:
        Tupla ``(estimated_error_pct, confidence_interval)``.
        Ambos podem ser ``None`` quando não há métricas disponíveis.
    """
    metrics = metadata.get("metrics", {}) if isinstance(metadata, dict) else {}

    mape_price = metrics.get("mape_price")
    rmse_price = metrics.get("rmse_price")

    estimated_error_pct: float | None = None
    if mape_price is not None:
        estimated_error_pct = float(mape_price)
    elif rmse_price is not None and predicted_price > 0:
        estimated_error_pct = float((float(rmse_price) / predicted_price) * 100)

    if rmse_price is not None:
        margin = Z_SCORE_95_CONFIDENCE * float(rmse_price)
    elif estimated_error_pct is not None:
        margin = predicted_price * (estimated_error_pct / 100)
    else:
        return estimated_error_pct, None

    ci = ConfidenceInterval(
        low_usd=round(max(0.0, predicted_price - margin), 2),
        high_usd=round(predicted_price + margin, 2),
    )
    return estimated_error_pct, ci


# ---------------------------------------------------------------------------
# InferenceService
# ---------------------------------------------------------------------------


class InferenceService:
    """Orquestra todo o pipeline de inferência LSTM: dados → features → modelo → IC.

    Depende de :class:`~src.domain.ports.LoadedArtifacts` e
    :class:`~src.domain.ports.MarketDataPort` injetados via construtor
    (princípio da inversão de dependência — SOLID D).

    Args:
        artifacts: Artefatos de ML carregados (modelo, scalers, metadados).
        market_data: Porta de dados de mercado (ex: ``_DownloadWithRetryPort``).
    """

    def __init__(
        self,
        artifacts: LoadedArtifacts,
        market_data: MarketDataPort,
    ) -> None:
        self._artifacts = artifacts
        self._market_data = market_data

    def predict(self, ticker: str, use_partial_candle: bool = False) -> InferenceResult:
        """Executa inferência completa para o próximo fechamento horário do ativo.

        Args:
            ticker: Símbolo do ativo (ex: ``"BTC-USD"``).
            use_partial_candle: Se ``True``, inclui o candle horário em formação
                na janela de entrada do modelo.

        Returns:
            :class:`InferenceResult` com preço previsto, timestamps e métricas
            de incerteza.

        Raises:
            :class:`DataServiceError`: Dados de mercado sem colunas obrigatórias.
            :class:`InsufficientDataError`: Dados insuficientes para a janela do modelo.
            RuntimeError: Todas as fontes de dados falharam (propagado da porta).
        """
        df, data_source = self._market_data.download(ticker)

        if "Close" not in df.columns:
            raise DataServiceError("Dados de mercado sem coluna Close")

        metadata = self._artifacts.metadata
        n_features = int(metadata.get("n_features", 1))
        scaler = self._artifacts.scaler
        scaler_return = self._artifacts.scaler_return
        model = self._artifacts.model

        close_series = df["Close"].dropna()
        if not use_partial_candle:
            close_series = remove_incomplete_hour_candle(close_series)

        required_points = LOOKBACK + 1
        if len(close_series) < required_points:
            raise InsufficientDataError(
                f"Dados insuficientes para janela de retorno ({required_points} closes)."
            )

        predicted_log_return: float
        last_close: float
        last_observed_ts: pd.Timestamp

        if n_features > 1:
            # ----------------------------------------------------------------
            # Caminho multi-feature (OHLCV + indicadores técnicos)
            # ----------------------------------------------------------------
            if not {"High", "Low", "Volume"}.issubset(df.columns):
                raise DataServiceError(
                    "Dados de mercado sem colunas OHLCV necessárias "
                    "para inferência multi-feature."
                )
            ohlcv = df[["Close", "High", "Low", "Volume"]].dropna()
            ohlcv = ohlcv.loc[close_series.index]

            features_df = build_feature_matrix(ohlcv)

            if len(features_df) < LOOKBACK:
                raise InsufficientDataError(
                    f"Dados insuficientes para janela multi-feature de {LOOKBACK}h."
                )

            window = features_df.to_numpy()[-LOOKBACK:]
            scaled_input = scaler.transform(window)
            x_input = scaled_input.reshape(1, LOOKBACK, n_features)

            predicted_scaled = model.predict(x_input, verbose=0)

            if scaler_return is not None:
                predicted_log_return = float(
                    scaler_return.inverse_transform(
                        predicted_scaled.reshape(-1, 1)
                    ).reshape(-1)[0]
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
            # ----------------------------------------------------------------
            # Caminho single-feature (log_return apenas — modelo legado)
            # ----------------------------------------------------------------
            log_price_series = pd.Series(
                np.log(close_series.values), index=close_series.index
            )
            return_series = log_price_series.diff().dropna()

            if len(return_series) < LOOKBACK:
                raise InsufficientDataError(
                    f"Dados insuficientes para gerar janela de retorno de {LOOKBACK}h."
                )

            last_returns = np.asarray(
                return_series.to_numpy()[-LOOKBACK:],
                dtype=float,
            ).reshape(-1, 1)
            scaled_input = scaler.transform(last_returns)
            x_input = scaled_input.reshape(1, LOOKBACK, 1)

            predicted_scaled = model.predict(x_input, verbose=0)
            predicted_log_return = float(
                scaler.inverse_transform(predicted_scaled).reshape(-1)[0]
            )
            last_close = float(close_series.iloc[-1])
            last_observed_ts = pd.Timestamp(close_series.index[-1])

        # ------------------------------------------------------------------
        # Conversão para preço absoluto e cálculo de incerteza
        # ------------------------------------------------------------------
        predicted_price = float(last_close * np.exp(predicted_log_return))
        estimated_error_pct, ci = estimate_uncertainty(predicted_price, metadata)

        logger.debug(
            "Inferência concluída: ticker=%s source=%s price=%.2f",
            ticker,
            data_source,
            predicted_price,
        )

        return InferenceResult(
            predicted_price_usd=predicted_price,
            last_close=last_close,
            last_observed_ts=last_observed_ts,
            data_source=data_source,
            estimated_error_pct=estimated_error_pct,
            confidence_interval=ci,
        )
