"""Groq, on its free tier."""

from __future__ import annotations

from typing import Final

from signaldesk.rag.llm.providers.openai_compatible import OpenAiCompatibleProvider

BASE_URL: Final[str] = "https://api.groq.com"
COMPLETIONS_URL: Final[str] = f"{BASE_URL}/openai/v1/chat/completions"

#: Free-tier models. Anything outside this set is refused at construction, so a
#: paid model cannot be reached by editing an environment variable.
ALLOWED_MODELS: Final[frozenset[str]] = frozenset(
    {
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
    }
)

DEFAULT_MODEL: Final[str] = "llama-3.3-70b-versatile"


class GroqProvider(OpenAiCompatibleProvider):
    name = "groq"
    endpoint = COMPLETIONS_URL

    def _api_key(self) -> str:
        return self.settings.groq_api_key
