"""Minimal compatibility layer for ragas import-time dependency on langchain_openai.llms.base."""

from __future__ import annotations


class BaseOpenAI:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs