"""Ollama, running locally. The last resort in the chain.

Gated on an explicit setting rather than on the presence of a base URL. Ollama
has no API key, so a URL-based check reports "configured" on every machine
including continuous integration, and the provider then sits permanently at the
end of the chain with nothing listening on it. Every exhausted chain would end
in a connection timeout instead of a fast skip, and the timeout would be charged
to whichever call happened to exhaust the chain.

``OLLAMA_ENABLED`` defaults to false. Unset means skipped, exactly like a
missing API key.
"""

from __future__ import annotations

from typing import Any, Final
from urllib.parse import urlparse

from signaldesk.core.config import Settings
from signaldesk.core.errors import ProviderError
from signaldesk.rag.llm.base import ChatRequest, HttpProvider, RawCompletion
from signaldesk.rag.llm.types import TokenUsage

DEFAULT_MODEL: Final[str] = "qwen2.5:7b-instruct"

#: Hosts a local Ollama may live on. The compose service name is included
#: because that is the address from inside the container. Anything else is a
#: remote endpoint that could bill, so it is refused at construction.
ALLOWED_HOSTS: Final[frozenset[str]] = frozenset(
    {"localhost", "127.0.0.1", "::1", "0.0.0.0", "ollama", "host.docker.internal"}
)


def host_is_local(base_url: str) -> bool:
    """Whether a base URL points at a loopback or compose-local Ollama."""
    try:
        parsed = urlparse(base_url)
    except ValueError:
        return False
    return (parsed.hostname or "") in ALLOWED_HOSTS


class OllamaProvider(HttpProvider):
    name = "ollama"

    def __init__(self, model: str, settings: Settings) -> None:
        super().__init__(model, settings)
        # Set here rather than as a class attribute: unlike the hosted
        # providers, where this lives is configuration.
        self.endpoint = settings.ollama_base_url

    @property
    def is_configured(self) -> bool:
        return bool(self.settings.ollama_enabled)

    def _url(self) -> str:
        return f"{self.settings.ollama_base_url.rstrip('/')}/api/chat"

    def _headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json"}

    def _body(self, chat: ChatRequest) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [
                {"role": message.role, "content": message.content} for message in chat.messages
            ],
            "stream": False,
            # Ollama takes the JSON Schema directly here and constrains decoding
            # to it, which is the closest thing in the chain to a guarantee.
            "format": chat.json_schema,
            "options": {
                "temperature": chat.temperature,
                "num_predict": chat.max_output_tokens,
            },
        }

    def _parse(self, payload: dict[str, Any]) -> RawCompletion:
        body = payload.get("message") or {}
        content = body.get("content")
        if not isinstance(content, str) or not content.strip():
            message = f"ollama returned an empty message: {str(payload)[:200]}"
            raise ProviderError(message)

        def _count(key: str) -> int | None:
            value = payload.get(key)
            return int(value) if isinstance(value, (int, float)) else None

        prompt = _count("prompt_eval_count")
        completion = _count("eval_count")
        total = None if prompt is None or completion is None else prompt + completion

        return RawCompletion(
            text=content,
            usage=TokenUsage(prompt=prompt, completion=completion, total=total),
            raw=payload,
        )
