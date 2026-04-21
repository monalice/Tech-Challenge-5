"""Minimal compatibility layer for ragas import-time dependency on langchain_openai.

The project uses Gemini via Google AI Studio. If this placeholder class is ever
instantiated directly, it raises to make the unsupported path explicit.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings


class OpenAIEmbeddings(Embeddings):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError(
            "OpenAIEmbeddings nao e suportado neste projeto. Use GoogleGenerativeAIEmbeddings com GOOGLE_API_KEY."
        )

    def embed_query(self, text: str) -> list[float]:
        raise RuntimeError(
            "OpenAIEmbeddings nao e suportado neste projeto. Use GoogleGenerativeAIEmbeddings com GOOGLE_API_KEY."
        )