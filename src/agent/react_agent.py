"""Agente ReAct com LangChain para orquestração das ferramentas de previsão Bitcoin.

Ferramentas disponíveis:
    - PrevisaoBitcoinTool   : executa o pipeline de inferência LSTM e retorna a previsão.
    - CotacaoAtualTool      : consulta a cotação atual do BTC via yfinance / Binance.
    - CryptoKnowledgeRAG    : recupera notícias e contexto cripto a partir de um vector store local.

Uso:
    from src.agent.react_agent import build_agent
    executor = build_agent(ml_artifacts)
    result   = executor.invoke({"input": "Qual a previsão do BTC para a próxima hora?"})
    logger.info(result["output"])
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, cast

import numpy as np
import pandas as pd
from langchain.agents import AgentExecutor, create_react_agent
from langchain_aws import ChatBedrock
from langchain.prompts import PromptTemplate
from langchain.tools import tool

try:
    from langfuse.callback import CallbackHandler as LangfuseHandler  # type: ignore[import-not-found]

    _LANGFUSE_AVAILABLE = True
except ImportError:
    LangfuseHandler = None
    _LANGFUSE_AVAILABLE = False

from src.agent.rag_pipeline import get_crypto_news_vector_store, similarity_search
from src.agent.llm_config import resolve_aws_region
from src.features.technical_features import build_feature_matrix
from src.infrastructure.market_data import BinanceSource, FallbackMarketData, YFinanceSource
from src.security.guardrails import InputGuardrail, OutputGuardrail
from src.domain.constants import (
    LOOKBACK,
    SUPPORTED_TICKER,
    Z_SCORE_95_CONFIDENCE,
)
from src.domain.time_utils import (
    remove_incomplete_hour_candle,
    remove_incomplete_hour_candle_df,
    timestamp_to_brt_iso,
    timestamp_to_utc_iso,
)

logger = logging.getLogger("stockcast.agent")
YFINANCE_MAX_RETRIES = 2

# Instância compartilhada da estratégia de mercado para o agente
_FALLBACK_MARKET_DATA: FallbackMarketData = FallbackMarketData(
    primary=YFinanceSource(),
    fallback=BinanceSource(),
    max_retries=YFINANCE_MAX_RETRIES,
)
DEFAULT_AGENT_LLM_MODEL = os.getenv(
    "AGENT_LLM_MODEL",
    os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-haiku-4-5-20251001-v1:0"),
)


def _get_env_optional_float(primary_key: str, fallback_key: str | None = None) -> float | None:
    """Lê uma variável de ambiente como float, com chave de fallback opcional.

    Args:
        primary_key: Nome da variável de ambiente principal.
        fallback_key: Nome da variável de ambiente de fallback, usada quando
            a principal está ausente ou vazia.

    Returns:
        Valor convertido para float, ou ``None`` se ambas as variáveis estiverem
        ausentes ou com valor em branco.
    """
    raw_value = os.getenv(primary_key)
    if (raw_value is None or raw_value.strip() == "") and fallback_key:
        raw_value = os.getenv(fallback_key)
    if raw_value is None or raw_value.strip() == "":
        return None
    return float(raw_value)


def _get_env_optional_int(primary_key: str, fallback_key: str | None = None) -> int | None:
    """Lê uma variável de ambiente como int, com chave de fallback opcional.

    Args:
        primary_key: Nome da variável de ambiente principal.
        fallback_key: Nome da variável de ambiente de fallback.

    Returns:
        Valor convertido para int, ou ``None`` se ambas as variáveis estiverem
        ausentes ou com valor em branco.
    """
    raw_value = os.getenv(primary_key)
    if (raw_value is None or raw_value.strip() == "") and fallback_key:
        raw_value = os.getenv(fallback_key)
    if raw_value is None or raw_value.strip() == "":
        return None
    return int(raw_value)


def _resolve_agent_temperature() -> float:
    """Resolve a temperatura do LLM a partir das variáveis de ambiente.

    Returns:
        Temperatura como float; padrão ``0.0`` quando não configurada.
    """
    value = _get_env_optional_float("AGENT_LLM_TEMPERATURE", "GEMINI_TEMPERATURE")
    return value if value is not None else 0.0


def _resolve_agent_top_p() -> float | None:
    """Resolve o parâmetro top-p do LLM a partir das variáveis de ambiente.

    Returns:
        Valor de top-p como float, ou ``None`` quando não configurado.
    """
    return _get_env_optional_float("AGENT_LLM_TOP_P", "GEMINI_TOP_P")


def _resolve_agent_top_k() -> int | None:
    """Resolve o parâmetro top-k do LLM a partir das variáveis de ambiente.

    Returns:
        Valor de top-k como int, ou ``None`` quando não configurado.
    """
    return _get_env_optional_int("AGENT_LLM_TOP_K", "GEMINI_TOP_K")


# ---------------------------------------------------------------------------
# Download de dados de mercado (delega para infrastructure.market_data)
# ---------------------------------------------------------------------------


def _download_market_data(ticker: str) -> tuple[pd.DataFrame, str]:
    """Baixa dados de mercado via FallbackMarketData (YFinance → Binance).

    Mantido como função de módulo para facilitar monkeypatch em testes.

    Args:
        ticker: Símbolo do ativo (ex: ``"BTC-USD"``).

    Returns:
        Tupla ``(DataFrame, source_name)``.

    Raises:
        RuntimeError: Se todas as fontes falharem.
    """
    return _FALLBACK_MARKET_DATA.download(ticker)


# ---------------------------------------------------------------------------
# Fábrica de ferramentas (encapsula ml_artifacts via closure)
# ---------------------------------------------------------------------------


def _make_tools(ml_artifacts: dict[str, Any]) -> list[Any]:
    """Cria e retorna as 3 ferramentas LangChain para o agente ReAct."""

    # ------------------------------------------------------------------
    # 1. PrevisaoBitcoinTool
    # ------------------------------------------------------------------
    @tool
    def previsao_bitcoin(query: str) -> str:  # noqa: ARG001
        """Executa o pipeline de inferência LSTM e retorna a previsão do próximo fechamento
        horário do Bitcoin (BTC-USD) em USD. Sempre chame esta ferramenta quando o usuário
        perguntar sobre previsão, forecast ou próximo preço do BTC.
        O parâmetro 'query' pode ser qualquer string — ela é ignorada internamente."""
        model = ml_artifacts.get("model")
        scaler = ml_artifacts.get("scaler")
        if model is None or scaler is None:
            return "Modelo não disponível. Os artefatos ainda não foram carregados."

        try:
            metadata = ml_artifacts.get("metadata", {})
            n_features = metadata.get("n_features", 1)
            scaler_return = ml_artifacts.get("scaler_return")

            df, data_source = _download_market_data(SUPPORTED_TICKER)

            if n_features > 1:
                if not {"High", "Low", "Volume"}.issubset(df.columns):
                    return "Dados OHLCV indisponíveis para inferência multi-feature."
                ohlcv = df[["Close", "High", "Low", "Volume"]].dropna()
                ohlcv = remove_incomplete_hour_candle_df(ohlcv)

                features_df = build_feature_matrix(ohlcv)

                if len(features_df) < LOOKBACK:
                    return (
                        "Dados insuficientes: "
                        f"{len(features_df)} candles disponíveis, necessário {LOOKBACK}."
                    )

                window = features_df.to_numpy()[-LOOKBACK:]
                scaled_input = scaler.transform(window)
                X_input = scaled_input.reshape(1, LOOKBACK, n_features)

                predicted_scaled = model.predict(X_input, verbose=0)

                if scaler_return is not None:
                    predicted_log_return = float(
                        scaler_return.inverse_transform(predicted_scaled.reshape(-1, 1)).reshape(
                            -1
                        )[0]
                    )
                else:
                    try:
                        min_val = float(scaler.data_min_[0])
                        max_val = float(scaler.data_max_[0])
                        predicted_log_return = (
                            float(predicted_scaled.reshape(-1)[0]) * (max_val - min_val) + min_val
                        )
                    except (AttributeError, IndexError):
                        predicted_log_return = float(predicted_scaled.reshape(-1)[0])

                last_close = float(ohlcv["Close"].iloc[-1])
                last_ts = pd.Timestamp(features_df.index[-1])
            else:
                close_series = df["Close"].dropna()
                close_series = remove_incomplete_hour_candle(close_series)
                log_price = pd.Series(np.log(close_series.values), index=close_series.index)
                ret_series = log_price.diff().dropna()

                if len(ret_series) < LOOKBACK:
                    return (
                        "Dados insuficientes: "
                        f"{len(ret_series)} candles disponíveis, necessário {LOOKBACK}."
                    )

                last_returns = np.asarray(ret_series.to_numpy()[-LOOKBACK:], dtype=float).reshape(
                    -1, 1
                )
                scaled_input = scaler.transform(last_returns)
                X_input = scaled_input.reshape(1, LOOKBACK, 1)

                predicted_scaled = model.predict(X_input, verbose=0)
                predicted_log_return = float(
                    scaler.inverse_transform(predicted_scaled).reshape(-1)[0]
                )
                last_close = float(close_series.iloc[-1])
                last_ts = pd.Timestamp(close_series.index[-1])

            predicted_price = last_close * np.exp(predicted_log_return)
            forecast_for_ts = last_ts + pd.Timedelta(hours=1)
            forecast_close_ts = forecast_for_ts + pd.Timedelta(hours=1) - pd.Timedelta(seconds=1)

            # Incerteza (MAPE/RMSE do metadata)
            metrics = metadata.get("metrics", {}) if isinstance(metadata, dict) else {}
            mape = metrics.get("mape_price")
            rmse = metrics.get("rmse_price")
            confidence_info = ""
            if rmse is not None:
                margin = Z_SCORE_95_CONFIDENCE * float(rmse)
                ci_low = max(0.0, predicted_price - margin)
                ci_high = predicted_price + margin
                confidence_info = f" | IC 95%: [{ci_low:,.2f} – {ci_high:,.2f}] USD"
            elif mape is not None:
                confidence_info = f" | erro estimado: {float(mape):.2f}%"

            result = (
                f"Previsão BTC-USD para {timestamp_to_brt_iso(forecast_for_ts)} (BRT): "
                f"**USD {predicted_price:,.2f}**{confidence_info}\n"
                f"Último candle usado: {timestamp_to_brt_iso(last_ts)} (BRT)\n"
                f"Fechamento previsto até: {timestamp_to_brt_iso(forecast_close_ts)} (BRT)\n"
                f"Fonte de dados: {data_source}"
            )
            logger.info("[agent:previsao_bitcoin] %s", result)
            return result

        except Exception as exc:
            logger.error("[agent:previsao_bitcoin] erro: %s", exc, exc_info=True)
            return f"Erro ao gerar previsão: {exc}"

    # ------------------------------------------------------------------
    # 2. CotacaoAtualTool
    # ------------------------------------------------------------------
    @tool
    def cotacao_atual(query: str) -> str:  # noqa: ARG001
        """Consulta a cotação atual do Bitcoin (BTC-USD) em tempo real via Yahoo Finance
        ou Binance (fallback automático). Retorna o preço de fechamento mais recente,
        variação nas últimas 24h e o volume negociado. Use esta ferramenta quando o
        usuário perguntar sobre o preço atual, cotação ou valor de mercado do BTC.
        O parâmetro 'query' pode ser qualquer string."""
        try:
            df, data_source = _download_market_data(SUPPORTED_TICKER)
            close_series = df["Close"].dropna()
            close_series = remove_incomplete_hour_candle(close_series)

            if len(close_series) < 2:
                return "Dados insuficientes para calcular cotação e variação."

            last_price = float(close_series.iloc[-1])
            prev_price = float(close_series.iloc[-2])
            price_change_pct = ((last_price - prev_price) / prev_price) * 100
            last_ts = pd.Timestamp(close_series.index[-1])

            # Volume (se disponível)
            volume_info = ""
            if "Volume" in df.columns:
                volume = df["Volume"].dropna()
                if len(volume) > 0:
                    last_vol = float(volume.iloc[-1])
                    volume_info = f"\nVolume último candle: {last_vol:,.2f} BTC"

            # Máxima e mínima das últimas 24h
            price_24h_info = ""
            recent_24h = close_series.iloc[-24:] if len(close_series) >= 24 else close_series
            high_24h = float(recent_24h.max())
            low_24h = float(recent_24h.min())
            price_24h_info = f"\nMáxima 24h: USD {high_24h:,.2f} | Mínima 24h: USD {low_24h:,.2f}"

            direction = "▲" if price_change_pct >= 0 else "▼"
            result = (
                f"Cotação BTC-USD:\n"
                f"Preço atual: **USD {last_price:,.2f}**\n"
                f"Variação vs candle anterior: {direction} {price_change_pct:+.2f}%"
                f"{price_24h_info}{volume_info}\n"
                f"Referência: {timestamp_to_brt_iso(last_ts)} (BRT)\n"
                f"Fonte: {data_source}"
            )
            logger.info(
                "[agent:cotacao_atual] preço=%.2f variação=%.2f%%", last_price, price_change_pct
            )
            return result

        except Exception as exc:
            logger.error("[agent:cotacao_atual] erro: %s", exc, exc_info=True)
            return f"Erro ao consultar cotação: {exc}"

    # ------------------------------------------------------------------
    # 3. CryptoKnowledgeRAG  (RAG com vector store local)
    # ------------------------------------------------------------------
    crypto_news_store = None
    crypto_news_error: Exception | None = None

    def _get_crypto_news_store() -> Any:
        nonlocal crypto_news_store, crypto_news_error
        if crypto_news_store is None and crypto_news_error is None:
            try:
                crypto_news_store = get_crypto_news_vector_store(backend="chroma")
            except Exception as exc:
                crypto_news_error = exc
                raise
        if crypto_news_error is not None:
            raise crypto_news_error
        return crypto_news_store

    @tool("CryptoKnowledgeRAG")
    def crypto_knowledge_rag(query: str) -> str:
        """Recupera notícias e contexto de mercado cripto a partir de um vector store local.
        Use esta ferramenta quando a pergunta envolver ETFs, macro, dominância, mineração,
        volatilidade ou notícias que possam contextualizar a previsão do BTC."""
        try:
            store = _get_crypto_news_store()
            docs = similarity_search(store, query, k=3)
        except Exception as exc:
            logger.error(
                "[agent:crypto_knowledge_rag] erro ao consultar vector store: %s",
                exc,
                exc_info=True,
            )
            return f"RAG indisponível no momento: {exc}"

        if not docs:
            return (
                "Nenhum contexto relevante foi encontrado no repositório vetorial "
                "de notícias cripto. Considere responder com cautela e explicitar "
                "a incerteza."
            )

        formatted_contexts: list[str] = []
        for index, doc in enumerate(docs, start=1):
            title = str(doc.metadata.get("title", "Notícia sem título"))
            topic = str(doc.metadata.get("topic", "geral"))
            published_at = str(doc.metadata.get("published_at", "data desconhecida"))
            formatted_contexts.append(
                f"[Contexto {index}] {title} | tópico: {topic} | "
                f"data: {published_at}\n{doc.page_content}"
            )

        logger.info(
            "[agent:crypto_knowledge_rag] query=%r → %d documentos recuperados", query, len(docs)
        )
        return "\n\n".join(formatted_contexts)

    return [previsao_bitcoin, cotacao_atual, crypto_knowledge_rag]


class _GuardedAgentExecutor:
    """Wrapper que aplica guardrails de entrada e saída em torno de um AgentExecutor.

    Intercepta a chamada a :meth:`invoke`, valida o prompt de entrada com
    :class:`~src.security.guardrails.InputGuardrail` e sanitiza a resposta com
    :class:`~src.security.guardrails.OutputGuardrail`. Delega todos os outros
    atributos ao executor base via ``__getattr__``.
    """

    def __init__(
        self,
        base_executor: AgentExecutor,
        input_guardrail: InputGuardrail,
        output_guardrail: OutputGuardrail,
    ) -> None:
        """Inicializa o wrapper com o executor base e os guardrails.

        Args:
            base_executor: AgentExecutor LangChain a ser protegido.
            input_guardrail: Instância de guardrail para validação de entrada.
            output_guardrail: Instância de guardrail para sanitização de saída.
        """
        self._base_executor = base_executor
        self._input_guardrail = input_guardrail
        self._output_guardrail = output_guardrail

    def __getattr__(self, name: str) -> Any:
        """Delega atributos desconhecidos ao executor base.

        Args:
            name: Nome do atributo solicitado.

        Returns:
            Atributo correspondente no executor base.
        """
        return getattr(self._base_executor, name)

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> dict[str, Any]:
        """Invoca o agente com guardrails de entrada e saída aplicados.

        Bloqueia entradas que contenham prompt injection ou context stuffing e
        sanitiza a saída antes de devolvê-la ao chamador.

        Args:
            input: Dicionário com chave ``"input"`` contendo o prompt do usuário.
            config: Configuração opcional repassada ao executor base.
            **kwargs: Argumentos adicionais repassados ao executor base.

        Returns:
            Dicionário com ``"output"`` (string sanitizada), ``"intermediate_steps"``
            e ``"guardrails"`` com metadados de validação. Quando a entrada é
            bloqueada, retorna imediatamente sem chamar o executor base.

        Raises:
            ValueError: Se *input* não for um dicionário.
        """
        if not isinstance(input, dict):
            raise ValueError("Entrada do agente deve ser um dicionário com a chave 'input'.")

        user_prompt = str(input.get("input", ""))
        validation = self._input_guardrail.validate(user_prompt)
        if not validation.allowed:
            return {
                "output": (
                    "Entrada bloqueada pela esteira de segurança: "
                    f"{validation.reason or 'prompt injection/context stuffing detectado'}"
                ),
                "intermediate_steps": [],
                "guardrails": {
                    "input_allowed": False,
                    "input_reason": validation.reason,
                    "output_sanitized": False,
                },
            }

        guarded_input = dict(input)
        guarded_input["input"] = validation.sanitized_text or user_prompt
        if config is None:
            result = self._base_executor.invoke(guarded_input, **kwargs)
        else:
            result = self._base_executor.invoke(guarded_input, config=config, **kwargs)

        if isinstance(result, dict) and isinstance(result.get("output"), str):
            result["output"] = self._output_guardrail.sanitize(result["output"])
            guardrails_meta = result.get("guardrails")
            if not isinstance(guardrails_meta, dict):
                guardrails_meta = {}
            guardrails_meta.update(
                {
                    "input_allowed": True,
                    "input_reason": None,
                    "output_sanitized": True,
                }
            )
            result["guardrails"] = guardrails_meta

        return result


# ---------------------------------------------------------------------------
# Prompt ReAct
# ---------------------------------------------------------------------------
_REACT_PROMPT_TEMPLATE = """Você é um assistente especialista em mercados de criptomoedas,
com acesso a dados de mercado em tempo real e um modelo LSTM para previsão do Bitcoin.

Responda sempre em português brasileiro. Seja preciso, objetivo e cite os dados
retornados pelas ferramentas.

Você tem acesso às seguintes ferramentas:

{tools}

Use o seguinte formato estritamente:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

_REACT_PROMPT = PromptTemplate.from_template(_REACT_PROMPT_TEMPLATE)


# ---------------------------------------------------------------------------
# Fábrica do agente
# ---------------------------------------------------------------------------


def build_agent(ml_artifacts: dict[str, Any]) -> AgentExecutor:
    """Constrói e retorna um AgentExecutor ReAct configurado com as 3 ferramentas.

    Requer uma região AWS configurada e credenciais válidas para Amazon Bedrock.

    Args:
        ml_artifacts: dicionário compartilhado com 'model', 'scaler', 'scaler_return' e 'metadata'.

    Returns:
        AgentExecutor pronto para receber ``{"input": "<pergunta>"}``
        via ``.invoke()``.
    """
    bedrock_region = resolve_aws_region()
    if not bedrock_region:
        raise OSError(
            "A região AWS para Amazon Bedrock não está definida. Use BEDROCK_AWS_REGION, AWS_REGION ou AWS_DEFAULT_REGION."
        )

    model_kwargs: dict[str, Any] = {"temperature": _resolve_agent_temperature()}
    top_p = _resolve_agent_top_p()
    if top_p is not None:
        model_kwargs["top_p"] = top_p
    top_k = _resolve_agent_top_k()
    if top_k is not None:
        model_kwargs["top_k"] = top_k

    llm = ChatBedrock(
        model_id=DEFAULT_AGENT_LLM_MODEL,
        region_name=bedrock_region,
        model_kwargs=model_kwargs,
    )

    tools = _make_tools(ml_artifacts)
    if len(tools) < 3:
        raise ValueError(
            "A arquitetura de referência exige no mínimo 3 tools customizadas para o agente ReAct."
        )

    agent = create_react_agent(llm=cast(Any, llm), tools=tools, prompt=_REACT_PROMPT)

    # Langfuse: telemetria de qualidade LLM (faithfulness, relevância, latência).
    # Só é ativado quando LANGFUSE_PUBLIC_KEY e LANGFUSE_SECRET_KEY estão definidas.
    callbacks: list[Any] = []
    pub_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    sec_key = os.getenv("LANGFUSE_SECRET_KEY")
    if _LANGFUSE_AVAILABLE and pub_key and sec_key:
        try:
            lf_host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
            if LangfuseHandler is None:
                raise RuntimeError("Langfuse indisponível")
            langfuse_handler = LangfuseHandler(
                public_key=pub_key,
                secret_key=sec_key,
                host=lf_host,
            )
            callbacks.append(langfuse_handler)
            logger.info("[agent] Langfuse callback ativado (host=%s)", lf_host)
        except Exception as exc:
            logger.warning("[agent] Falha ao inicializar Langfuse: %s", exc)

    base_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        callbacks=callbacks if callbacks else None,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=int(os.getenv("AGENT_MAX_ITERATIONS", "6")),
        return_intermediate_steps=True,
    )

    input_guardrail = InputGuardrail(max_input_chars=InputGuardrail.MAX_INPUT_CHARS)
    output_guardrail = OutputGuardrail()
    return cast(
        AgentExecutor,
        _GuardedAgentExecutor(
            base_executor=base_executor,
            input_guardrail=input_guardrail,
            output_guardrail=output_guardrail,
        ),
    )
