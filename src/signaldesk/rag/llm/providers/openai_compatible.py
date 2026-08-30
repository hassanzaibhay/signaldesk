"""The chat-completions shape, which Groq and Cerebras both speak.

Two providers, one wire format, so the request building and response parsing
live here once. Only the base URL, the key and the model list differ, and those
are what the two subclasses supply.

Refusals arrive two ways in this format and both are answers rather than
failures: an explicit ``refusal`` field on the message, and a
``finish_reason`` of ``content_filter``. Neither is a reason to try the next
provider - the model has told the caller something.
"""

from __future__ import annotations

from typing import Any

from signaldesk.core.errors import ProviderError
from signaldesk.rag.llm.base import ChatRequest, HttpProvider, RawCompletion, usage_from
from signaldesk.rag.llm.types import Refusal, TokenUsage


class OpenAiCompatibleProvider(HttpProvider):
    """A provider speaking OpenAI's chat-completions API."""

    #: Set by the subclass. The full completions URL.
    endpoint: str = ""

    def _api_key(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key())

    def _url(self) -> str:
        return self.endpoint

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key()}",
            "Content-Type": "application/json",
        }

    def _body(self, chat: ChatRequest) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [
                {"role": message.role, "content": message.content} for message in chat.messages
            ],
            "temperature": chat.temperature,
            "max_tokens": chat.max_output_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": chat.schema_name,
                    "schema": chat.json_schema,
                    "strict": True,
                },
            },
        }

    def _parse(self, payload: dict[str, Any]) -> RawCompletion:
        choices = payload.get("choices") or []
        if not choices:
            message = f"{self.name} returned no choices: {str(payload)[:200]}"
            raise ProviderError(message)

        choice = choices[0]
        usage_block = payload.get("usage")
        usage = (
            usage_from(
                usage_block,
                prompt="prompt_tokens",
                completion="completion_tokens",
                total="total_tokens",
            )
            if isinstance(usage_block, dict)
            else TokenUsage()
        )

        body = choice.get("message") or {}
        refusal_text = body.get("refusal")
        if isinstance(refusal_text, str) and refusal_text.strip():
            return RawCompletion(
                usage=usage,
                refusal=Refusal(reason=refusal_text.strip(), provider_code="refusal"),
                raw=payload,
            )

        if choice.get("finish_reason") == "content_filter":
            return RawCompletion(
                usage=usage,
                refusal=Refusal(
                    reason=f"{self.name} stopped on its content filter",
                    provider_code="content_filter",
                ),
                raw=payload,
            )

        content = body.get("content")
        if not isinstance(content, str) or not content.strip():
            message = f"{self.name} returned an empty message: {str(payload)[:200]}"
            raise ProviderError(message)

        return RawCompletion(text=content, usage=usage, raw=payload)
