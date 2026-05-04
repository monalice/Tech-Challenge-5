"""Ports (interfaces) do domínio — contratos para injeção de dependência (Clean Architecture).

Define os Protocols que desacoplam a camada de domínio das implementações concretas
de infraestrutura (fontes de mercado, LLMs) e a dataclass tipada que carrega os
artefatos de ML em memória.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class MarketDataPort(Protocol):
    """Contrato para fontes de dados de mercado."""

    def download(self, ticker: str) -> tuple[pd.DataFrame, str]:
        """Baixa dados de mercado para o ticker informado.

        Args:
            ticker: Símbolo do ativo (ex: ``"BTC-USD"``).

        Returns:
            Tupla ``(DataFrame, source_name)`` onde *source_name* identifica
            a origem dos dados (ex: ``"yfinance"``, ``"binance"``).
        """
        ...


@runtime_checkable
class LLMPort(Protocol):
    """Contrato para executores de LLM (agentes ReAct, chains, etc.)."""

    def invoke(self, input: dict[str, Any], config: Any = None, **kwargs: Any) -> dict[str, Any]:
        """Executa o LLM com a entrada fornecida.

        Args:
            input: Dicionário de entrada para o modelo.
            config: Configuração opcional de execução.
            **kwargs: Argumentos adicionais de execução.

        Returns:
            Dicionário com a resposta e metadados do LLM.
        """
        ...


@dataclass
class LoadedArtifacts:
    """Artefatos de ML carregados em memória — modelo, scalers e metadados."""

    model: Any
    scaler: Any
    scaler_return: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Retorna representação dict compatível com a interface legada ``build_agent()``.

        Returns:
            Dicionário com chaves ``model``, ``scaler``, ``scaler_return`` e ``metadata``.
        """
        return {
            "model": self.model,
            "scaler": self.scaler,
            "scaler_return": self.scaler_return,
            "metadata": self.metadata,
        }
