from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.agent import react_agent


class _DummyModel:
    def predict(self, values, verbose=0):  # noqa: ARG002
        return np.full((len(values), 1), 0.01, dtype=float)


class _DummyScaler:
    data_min_ = np.array([-0.1], dtype=float)
    data_max_ = np.array([0.1], dtype=float)

    def transform(self, values):
        return np.asarray(values, dtype=float)

    def inverse_transform(self, values):
        return np.asarray(values, dtype=float)


class _DummyDoc:
    def __init__(self, title: str, topic: str, published_at: str, content: str) -> None:
        self.metadata = {
            "title": title,
            "topic": topic,
            "published_at": published_at,
        }
        self.page_content = content


def _mock_market_df(periods: int = 120) -> pd.DataFrame:
    end = pd.Timestamp.utcnow().floor("h") - pd.Timedelta(hours=1)
    index = pd.date_range(end=end, periods=periods, freq="h", tz="UTC")
    close = np.linspace(100_000, 101_000, periods)
    high = close + 50
    low = close - 50
    volume = np.full(periods, 120.0)
    return pd.DataFrame(
        {"Close": close, "High": high, "Low": low, "Volume": volume},
        index=index,
    )


def _find_tool(tools: list, name: str):
    return next(tool for tool in tools if tool.name == name)


def test_download_market_data_uses_binance_fallback(monkeypatch):
    monkeypatch.setattr(react_agent, "_fetch_yfinance", lambda ticker: (_ for _ in ()).throw(ValueError("yf down")))
    monkeypatch.setattr(react_agent, "_fetch_binance", lambda limit=200: _mock_market_df())

    df, source = react_agent._download_market_data("BTC-USD")

    assert source == "binance"
    assert not df.empty


def test_make_tools_returns_three_tools_and_handles_missing_artifacts():
    tools = react_agent._make_tools({})

    assert len(tools) == 3

    previsao_tool = _find_tool(tools, "previsao_bitcoin")
    output = previsao_tool.invoke("Qual a previsão?")

    assert "Modelo não disponível" in output


def test_tools_execute_with_mocks_and_return_expected_sections(monkeypatch):
    ml_artifacts = {
        "model": _DummyModel(),
        "scaler": _DummyScaler(),
        "metadata": {"n_features": 1, "metrics": {"rmse_price": 100.0}},
    }

    monkeypatch.setattr(
        react_agent,
        "_download_market_data",
        lambda ticker: (_mock_market_df(), "mock"),
    )
    monkeypatch.setattr(react_agent, "get_crypto_news_vector_store", lambda backend="chroma": object())
    monkeypatch.setattr(
        react_agent,
        "similarity_search",
        lambda store, query, k=3: [  # noqa: ARG005
            _DummyDoc(
                title="ETF com fluxo positivo",
                topic="etfs",
                published_at="2026-04-20",
                content="Fluxo institucional segue resiliente.",
            )
        ],
    )

    tools = react_agent._make_tools(ml_artifacts)
    previsao_tool = _find_tool(tools, "previsao_bitcoin")
    cotacao_tool = _find_tool(tools, "cotacao_atual")
    rag_tool = _find_tool(tools, "CryptoKnowledgeRAG")

    previsao = previsao_tool.invoke("Preveja o BTC")
    cotacao = cotacao_tool.invoke("Preço atual")
    contexto = rag_tool.invoke("Me dê contexto de ETFs")

    assert "Previsão BTC-USD" in previsao
    assert "Fonte de dados: mock" in previsao
    assert "Cotação BTC-USD" in cotacao
    assert "Fonte: mock" in cotacao
    assert "[Contexto 1]" in contexto


def test_build_agent_requires_google_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(OSError, match="GOOGLE_API_KEY"):
        react_agent.build_agent({})


def test_build_agent_constructs_executor_with_three_tools(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    monkeypatch.setattr(react_agent, "ChatGoogleGenerativeAI", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(react_agent, "create_react_agent", lambda llm, tools, prompt: "fake-agent")
    monkeypatch.setattr(
        react_agent,
        "_make_tools",
        lambda artifacts: [SimpleNamespace(name="a"), SimpleNamespace(name="b"), SimpleNamespace(name="c")],
    )

    class _FakeExecutor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(react_agent, "AgentExecutor", _FakeExecutor)

    executor: Any = react_agent.build_agent({})

    assert len(executor.kwargs["tools"]) == 3
    assert executor.kwargs["max_iterations"] == 6
    assert executor.kwargs["return_intermediate_steps"] is True


def test_agent_response_can_combine_forecast_with_rag_context(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    ml_artifacts = {
        "model": _DummyModel(),
        "scaler": _DummyScaler(),
        "metadata": {"n_features": 1, "metrics": {"mape_price": 2.0}},
    }

    monkeypatch.setattr(react_agent, "ChatGoogleGenerativeAI", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(react_agent, "create_react_agent", lambda llm, tools, prompt: "fake-agent")
    monkeypatch.setattr(
        react_agent,
        "_download_market_data",
        lambda ticker: (_mock_market_df(), "mock"),
    )
    monkeypatch.setattr(react_agent, "get_crypto_news_vector_store", lambda backend="chroma": object())
    monkeypatch.setattr(
        react_agent,
        "similarity_search",
        lambda store, query, k=3: [  # noqa: ARG005
            _DummyDoc(
                title="Risco macro em alta",
                topic="macro",
                published_at="2026-04-21",
                content="Juros reais influenciam o apetite por risco.",
            )
        ],
    )

    class _FakeExecutor:
        def __init__(self, **kwargs):
            self.tools = kwargs["tools"]

        def invoke(self, payload):
            query = payload["input"]
            previsao_tool = _find_tool(self.tools, "previsao_bitcoin")
            rag_tool = _find_tool(self.tools, "CryptoKnowledgeRAG")
            previsao = previsao_tool.invoke(query)
            contexto = rag_tool.invoke(query)
            return {"output": f"{previsao}\n\nContexto:\n{contexto}"}

    monkeypatch.setattr(react_agent, "AgentExecutor", _FakeExecutor)

    executor = react_agent.build_agent(ml_artifacts)
    result = executor.invoke({"input": "Qual a previsão e o contexto?"})

    assert "Previsão BTC-USD" in result["output"]
    assert "Contexto:" in result["output"]
    assert "[Contexto 1]" in result["output"]
