"""Cerebras, on its free tier."""

from __future__ import annotations

from typing import Final

from signaldesk.rag.llm.providers.openai_compatible import OpenAiCompatibleProvider

BASE_URL: Final[str] = "https://api.cerebras.ai"
COMPLETIONS_URL: Final[str] = f"{BASE_URL}/v1/chat/completions"

ALLOWED_MODELS: Final[frozenset[str]] = frozenset(
    {
        "llama-3.3-70b",
        "llama3.1-8b",
        "gpt-oss-120b",
        "qwen-3-32b",
    }
)

DEFAULT_MODEL: Final[str] = "llama-3.3-70b"


class CerebrasProvider(OpenAiCompatibleProvider):
    name = "cerebras"
    endpoint = COMPLETIONS_URL

    def _api_key(self) -> str:
        return self.settings.cerebras_api_key
