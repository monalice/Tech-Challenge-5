"""RAG pipeline utilities backed by Google embeddings and a local vector store."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
import os
from pathlib import Path
from typing import Any, Literal

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

DEFAULT_EMBEDDING_MODEL = os.getenv(
    "RAG_EMBEDDING_MODEL",
    os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004"),
)
DEFAULT_CHROMA_DIR = Path("data/processed/crypto_news_chroma")
VectorStoreBackend = Literal["chroma", "faiss"]

SIMULATED_CRYPTO_NEWS: list[dict[str, str]] = [
    {
        "title": "ETFs de BTC mantêm fluxo positivo na abertura da semana",
        "topic": "etfs",
        "published_at": "2026-04-18",
        "content": (
            "Os ETFs spot de Bitcoin nos Estados Unidos registraram nova rodada "
            "de entradas líquidas, sinalizando demanda institucional resiliente. "
            "Analistas destacam que fluxo consistente em ETFs costuma reduzir "
            "pressão vendedora de curto prazo e sustentar o sentimento comprador."
        ),
    },
    {
        "title": "Fed reforça cautela e mercado recalibra ativos de risco",
        "topic": "macro",
        "published_at": "2026-04-17",
        "content": (
            "Declarações de dirigentes do Federal Reserve levaram o mercado a "
            "revisar a trajetória esperada de cortes de juros. Para o Bitcoin, "
            "juros reais mais altos no curto prazo podem aumentar a volatilidade "
            "e reduzir o apetite por risco, embora a tese estrutural siga intacta."
        ),
    },
    {
        "title": "Mineradores reduzem vendas e aliviam pressão sobre o mercado",
        "topic": "mining",
        "published_at": "2026-04-15",
        "content": (
            "Dados on-chain sugerem desaceleração nas vendas de BTC por parte de "
            "mineradores. Menor pressão de distribuição pode favorecer estabilidade "
            "de preço no curto prazo, especialmente quando combinada com liquidez "
            "saudável nas corretoras."
        ),
    },
    {
        "title": "Volatilidade implícita sobe antes de divulgação macro nos EUA",
        "topic": "volatility",
        "published_at": "2026-04-14",
        "content": (
            "O mercado de opções de Bitcoin precificou maior volatilidade implícita "
            "antes de indicadores macroeconômicos relevantes. Em ambientes assim, "
            "previsões de curto prazo exigem mais cautela e devem comunicar faixas "
            "prováveis e risco de reversão rápida."
        ),
    },
    {
        "title": "Dominância do BTC permanece elevada enquanto altcoins perdem força",
        "topic": "dominance",
        "published_at": "2026-04-13",
        "content": (
            "A dominância do Bitcoin continua elevada, indicando que o capital "
            "segue concentrado no ativo principal do mercado cripto. Esse padrão "
            "costuma aparecer em momentos de busca por liquidez, qualidade e menor "
            "tolerância ao risco."
        ),
    },
]


def build_google_embeddings(
    model: str = DEFAULT_EMBEDDING_MODEL,
) -> GoogleGenerativeAIEmbeddings:
    """Create the Google embedding client used by the RAG pipeline."""
    return GoogleGenerativeAIEmbeddings(model=model)


def build_documents(
    texts: Sequence[str], metadatas: Sequence[dict[str, Any]] | None = None
) -> list[Document]:
    """Convert raw texts and optional metadata into LangChain documents."""
    if metadatas and len(texts) != len(metadatas):
        raise ValueError("texts e metadatas devem ter o mesmo tamanho.")

    documents: list[Document] = []
    for index, text in enumerate(texts):
        metadata = metadatas[index] if metadatas else {}
        documents.append(Document(page_content=text, metadata=metadata))
    return documents


def _import_chroma() -> Any:
    try:
        from langchain_community.vectorstores import Chroma
    except ImportError as exc:
        raise ImportError(
            "Chroma nao esta disponivel. "
            "Instale as dependencias de vector store em requirements.txt."
        ) from exc
    return Chroma


def _import_faiss() -> Any:
    try:
        from langchain_community.vectorstores import FAISS
    except ImportError as exc:
        raise ImportError(
            "FAISS nao esta disponivel. "
            "Instale as dependencias de vector store em requirements.txt."
        ) from exc
    return FAISS


def build_vector_store(
    documents: Sequence[Document],
    *,
    backend: VectorStoreBackend = "chroma",
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    persist_directory: str | Path | None = None,
) -> Any:
    """Build a Chroma or FAISS vector store using Google embeddings."""
    embeddings = build_google_embeddings(model=embedding_model)
    docs = list(documents)

    if backend == "chroma":
        Chroma = _import_chroma()
        kwargs: dict[str, object] = {}
        if persist_directory is not None:
            kwargs["persist_directory"] = str(persist_directory)
        return Chroma.from_documents(docs, embedding=embeddings, **kwargs)

    if backend == "faiss":
        FAISS = _import_faiss()
        return FAISS.from_documents(docs, embedding=embeddings)

    raise ValueError(f"Backend de vector store nao suportado: {backend}")


def build_vector_store_from_texts(
    texts: Sequence[str],
    *,
    metadatas: Sequence[dict[str, Any]] | None = None,
    backend: VectorStoreBackend = "chroma",
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    persist_directory: str | Path | None = None,
) -> Any:
    """Convenience helper to build a vector store directly from text chunks."""
    documents = build_documents(texts=texts, metadatas=metadatas)
    return build_vector_store(
        documents,
        backend=backend,
        embedding_model=embedding_model,
        persist_directory=persist_directory,
    )


def similarity_search(
    vector_store: Any,
    query: str,
    *,
    k: int = 4,
) -> list[Document]:
    """Run a similarity search using the configured vector store."""
    return list(vector_store.similarity_search(query, k=k))


def build_simulated_news_documents() -> list[Document]:
    """Build LangChain documents for the simulated crypto-news corpus."""
    return build_documents(
        texts=[item["content"] for item in SIMULATED_CRYPTO_NEWS],
        metadatas=[
            {
                "title": item["title"],
                "topic": item["topic"],
                "published_at": item["published_at"],
            }
            for item in SIMULATED_CRYPTO_NEWS
        ],
    )


@lru_cache(maxsize=2)
def get_crypto_news_vector_store(
    backend: VectorStoreBackend = "chroma",
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    persist_directory: str | None = None,
) -> Any:
    """Create and cache the local vector store used by the agent RAG tool."""
    documents = build_simulated_news_documents()
    resolved_directory = persist_directory or str(DEFAULT_CHROMA_DIR)
    return build_vector_store(
        documents,
        backend=backend,
        embedding_model=embedding_model,
        persist_directory=resolved_directory if backend == "chroma" else None,
    )
