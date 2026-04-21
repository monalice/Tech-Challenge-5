"""Compatibility shim for libraries that still import langchain_openai.

This repository no longer uses OpenAI directly. The shim exists only because
the installed ragas package imports ``langchain_openai.embeddings.OpenAIEmbeddings``
at module import time, even when a custom Gemini-backed embeddings model is used.
"""

from .chat_models import AzureChatOpenAI, ChatOpenAI
from .embeddings import OpenAIEmbeddings
from .llms import AzureOpenAI, OpenAI

__all__ = ["AzureChatOpenAI", "AzureOpenAI", "ChatOpenAI", "OpenAI", "OpenAIEmbeddings"]