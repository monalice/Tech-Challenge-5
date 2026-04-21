"""Minimal compatibility layer for ragas import-time dependency on langchain_openai.llms."""

from .base import BaseOpenAI


class OpenAI(BaseOpenAI):
    pass


class AzureOpenAI(OpenAI):
    pass


__all__ = ["AzureOpenAI", "BaseOpenAI", "OpenAI"]