"""Agente ReAct com LangChain para orquestração das ferramentas de previsão Bitcoin.

Ferramentas disponíveis:
    - PrevisaoBitcoinTool   : executa o pipeline de inferência LSTM e retorna a previsão.
    - CotacaoAtualTool      : consulta a cotação atual do BTC via yfinance / Binance.
    - CryptoKnowledgeRAG    : recupera notícias e contexto cripto a partir de um vector store local.

Uso:
    from src.agent.react_agent import build_agent
    executor = build_agent(ml_artifacts)
    result   = executor.invoke({"input": "Qual a previsão do BTC para a próxima hora?"})
    print(result["output"])
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

try:
    from langfuse.callback import CallbackHandler as LangfuseHandler

    _LANGFUSE_AVAILABLE = True
except ImportError:
    _LANGFUSE_AVAILABLE = False

from src.agent.rag_pipeline import get_crypto_news_vector_store, similarity_search

logger = logging.getLogger("stockcast.agent")

# ---------------------------------------------------------------------------
# Constantes (espelham app.py para manter consistência)
# ---------------------------------------------------------------------------
LOOKBACK = 60
SUPPORTED_TICKER = "BTC-USD"
BRASILIA_TZ = "America/Sao_Paulo"
BINANCE_SYMBOL = "BTCUSDT"
BINANCE_API_URL = "https://api.binance.com/api/v3/klines"
BINANCE_TIMEOUT_SECONDS = 10
YFINANCE_TIMEOUT_SECONDS = 10
YFINANCE_MAX_RETRIES = 2

# ---------------------------------------------------------------------------
# Indicadores técnicos (replicados de app.py para inferência independente)
# ---------------------------------------------------------------------------


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50.0)


def _compute_macd_signal(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.Series:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return ((macd_line - signal_line) / series.replace(0, np.nan)).fillna(0.0)


def _compute_bollinger_pct_b(
    series: pd.Series, period: int = 20, num_std: float = 2.0
) -> pd.Series:
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    band_width = (upper - lower).replace(0, np.nan)
    return ((series - lower) / band_width).fillna(0.5).clip(0.0, 1.0)


def _compute_sma_ratio(series: pd.Series, short: int = 7, long: int = 21) -> pd.Series:
    sma_short = series.rolling(window=short).mean()
    sma_long = series.rolling(window=long).mean()
    return ((sma_short / sma_long.replace(0, np.nan)) - 1.0).fillna(0.0)


def _compute_volume_ratio(volume: pd.Series, period: int = 24) -> pd.Series:
    vol_sma = volume.rolling(window=period).mean()
    return (volume / vol_sma.replace(0, np.nan)).fillna(1.0).clip(0.0, 10.0)


def _remove_incomplete_hour_candle(series: pd.Series) -> pd.Series:
    if len(series) < 2:
        return series
    last_ts = pd.Timestamp(series.index[-1])
    now_utc = pd.Timestamp.utcnow()
    now_ref = (
        now_utc.tz_localize(None) if last_ts.tzinfo is None else now_utc.tz_convert(last_ts.tz)
    )
    if last_ts >= now_ref.floor("h"):
        return series.iloc[:-1]
    return series


def _ts_to_utc_iso(ts: pd.Timestamp) -> str:
    ts = pd.Timestamp(ts)
    ts_utc = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts_utc.isoformat()


def _ts_to_brt_iso(ts: pd.Timestamp) -> str:
    ts = pd.Timestamp(ts)
    ts_utc = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts_utc.tz_convert(ZoneInfo(BRASILIA_TZ)).isoformat()


# ---------------------------------------------------------------------------
# Download de dados de mercado (yfinance → Binance fallback)
# ---------------------------------------------------------------------------


def _fetch_yfinance(ticker: str) -> pd.DataFrame:
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
    return df


def _fetch_binance(limit: int = 200) -> pd.DataFrame:
    resp = requests.get(
        BINANCE_API_URL,
        params={"symbol": BINANCE_SYMBOL, "interval": "1h", "limit": limit},
        timeout=BINANCE_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    raw = resp.json()
    if not raw:
        raise ValueError("Resposta vazia da Binance")
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


def _download_market_data(ticker: str) -> tuple[pd.DataFrame, str]:
    """Tenta Yahoo Finance; em caso de falha, usa Binance."""
    last_err: Exception | None = None
    for attempt in range(1, YFINANCE_MAX_RETRIES + 1):
        try:
            df = _fetch_yfinance(ticker)
            return df, "yfinance"
        except Exception as exc:
            last_err = exc
            logger.warning(
                "[agent] yfinance tentativa %d/%d falhou: %s", attempt, YFINANCE_MAX_RETRIES, exc
            )
            if attempt < YFINANCE_MAX_RETRIES:
                time.sleep(0.5 * attempt)

    logger.warning("[agent] Fallback para Binance...")
    try:
        return _fetch_binance(), "binance"
    except Exception as exc:
        raise RuntimeError(
            f"Todas as fontes de dados falharam. Último erro yfinance: {last_err}. Binance: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Fábrica de ferramentas (encapsula ml_artifacts via closure)
# ---------------------------------------------------------------------------


def _make_tools(ml_artifacts: dict[str, Any]) -> list:
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
                ohlcv = _remove_incomplete_hour_candle_df(ohlcv)

                close_col = ohlcv["Close"]
                log_return = np.log(close_col).diff()
                rsi = _compute_rsi(close_col) / 100.0
                macd_sig = _compute_macd_signal(close_col)
                bb_pct = _compute_bollinger_pct_b(close_col)
                sma_ratio = _compute_sma_ratio(close_col)
                vol_ratio = _compute_volume_ratio(ohlcv["Volume"])

                features_df = pd.DataFrame(
                    {
                        "log_return": log_return,
                        "rsi": rsi,
                        "macd_signal": macd_sig,
                        "bb_pct_b": bb_pct,
                        "sma_ratio": sma_ratio,
                        "vol_ratio": vol_ratio,
                    },
                    index=ohlcv.index,
                ).dropna()

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
                close_series = _remove_incomplete_hour_candle(close_series)
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
                margin = 1.96 * float(rmse)
                ci_low = max(0.0, predicted_price - margin)
                ci_high = predicted_price + margin
                confidence_info = f" | IC 95%: [{ci_low:,.2f} – {ci_high:,.2f}] USD"
            elif mape is not None:
                confidence_info = f" | erro estimado: {float(mape):.2f}%"

            result = (
                f"Previsão BTC-USD para {_ts_to_brt_iso(forecast_for_ts)} (BRT): "
                f"**USD {predicted_price:,.2f}**{confidence_info}\n"
                f"Último candle usado: {_ts_to_brt_iso(last_ts)} (BRT)\n"
                f"Fechamento previsto até: {_ts_to_brt_iso(forecast_close_ts)} (BRT)\n"
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
            close_series = _remove_incomplete_hour_candle(close_series)

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
                f"Referência: {_ts_to_brt_iso(last_ts)} (BRT)\n"
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

    def _get_crypto_news_store():
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


def _remove_incomplete_hour_candle_df(df: pd.DataFrame) -> pd.DataFrame:
    """Remove a última linha do DataFrame se ela corresponder ao candle horário em formação."""
    if len(df) < 2:
        return df
    last_ts = pd.Timestamp(df.index[-1])
    now_utc = pd.Timestamp.utcnow()
    now_ref = (
        now_utc.tz_localize(None) if last_ts.tzinfo is None else now_utc.tz_convert(last_ts.tz)
    )
    if last_ts >= now_ref.floor("h"):
        return df.iloc[:-1]
    return df


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

    Requer a variável de ambiente GOOGLE_API_KEY.

    Args:
        ml_artifacts: dicionário compartilhado com 'model', 'scaler', 'scaler_return' e 'metadata'.

    Returns:
        AgentExecutor pronto para receber ``{"input": "<pergunta>"}``
        via ``.invoke()``.
    """
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise OSError("A variável GOOGLE_API_KEY não está definida.")

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)

    tools = _make_tools(ml_artifacts)

    agent = create_react_agent(llm=llm, tools=tools, prompt=_REACT_PROMPT)

    # Langfuse: telemetria de qualidade LLM (faithfulness, relevância, latência).
    # Só é ativado quando LANGFUSE_PUBLIC_KEY e LANGFUSE_SECRET_KEY estão definidas.
    callbacks: list[Any] = []
    pub_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    sec_key = os.getenv("LANGFUSE_SECRET_KEY")
    if _LANGFUSE_AVAILABLE and pub_key and sec_key:
        try:
            lf_host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
            langfuse_handler = LangfuseHandler(  # type: ignore[reportPossiblyUnbound]
                public_key=pub_key,
                secret_key=sec_key,
                host=lf_host,
            )
            callbacks.append(langfuse_handler)
            logger.info("[agent] Langfuse callback ativado (host=%s)", lf_host)
        except Exception as exc:
            logger.warning("[agent] Falha ao inicializar Langfuse: %s", exc)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        callbacks=callbacks if callbacks else None,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=int(os.getenv("AGENT_MAX_ITERATIONS", "6")),
        return_intermediate_steps=True,
    )
