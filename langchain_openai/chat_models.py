"""Minimal compatibility layer for ragas import-time dependency on langchain_openai.

The project uses Gemini instead of OpenAI. These classes are placeholders so
third-party libraries that import them at module load time can still be imported.
"""

from __future__ import annotations


class ChatOpenAI:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class AzureChatOpenAI(ChatOpenAI):
    pass