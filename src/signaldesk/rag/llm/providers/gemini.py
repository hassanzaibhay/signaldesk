"""Gemini, on its free tier.

Different enough from the chat-completions shape to be its own module: roles
are `user` and `model` rather than `user` and `assistant`, the system prompt is
a separate field rather than a message, content is a list of parts, and the
schema goes in `generationConfig` rather than beside the messages.

Two refusal shapes, and both are answers rather than failures. A prompt blocked
before generation returns `promptFeedback.blockReason` and no candidate at all;
a generation stopped part way returns a candidate with a `finishReason` of
`SAFETY` or `RECITATION`. Reporting the first as a provider fault would send the
router shopping down the chain for a model that would say it.
"""

from __future__ import annotations

from typing import Any, Final

from signaldesk.core.errors import ProviderError
from signaldesk.rag.llm.base import ChatRequest, HttpProvider, RawCompletion, usage_from
from signaldesk.rag.llm.types import Refusal, TokenUsage

BASE_URL: Final[str] = "https://generativelanguage.googleapis.com"

ALLOWED_MODELS: Final[frozenset[str]] = frozenset(
    {
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
    }
)

DEFAULT_MODEL: Final[str] = "gemini-2.5-flash"

#: finishReason values that mean the model declined rather than failed.
REFUSAL_REASONS: Final[frozenset[str]] = frozenset(
    {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT"}
)

#: JSON Schema keywords Gemini's responseSchema does not accept. Sending them is
#: a 400, so they are stripped rather than passed through: the response is
#: validated against the caller's model afterwards regardless, so nothing is
#: lost by asking for a looser constraint than the model can express.
UNSUPPORTED_SCHEMA_KEYS: Final[frozenset[str]] = frozenset(
    {"$schema", "$defs", "$ref", "additionalProperties", "definitions", "title", "default"}
)


def _strip_schema(node: object) -> object:
    """Recursively drop keywords Gemini rejects."""
    if isinstance(node, dict):
        return {
            key: _strip_schema(value)
            for key, value in node.items()
            if key not in UNSUPPORTED_SCHEMA_KEYS
        }
    if isinstance(node, list):
        return [_strip_schema(item) for item in node]
    return node


class GeminiProvider(HttpProvider):
    name = "gemini"
    endpoint = BASE_URL

    def _api_key(self) -> str:
        return self.settings.gemini_api_key

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key())

    def _url(self) -> str:
        return f"{BASE_URL}/v1beta/models/{self.model}:generateContent"

    def _headers(self) -> dict[str, str]:
        # The key goes in a header rather than the query string so it does not
        # end up in a URL that gets logged or cached.
        return {"x-goog-api-key": self._api_key(), "Content-Type": "application/json"}

    def _body(self, chat: ChatRequest) -> dict[str, Any]:
        contents: list[dict[str, Any]] = []
        system: list[dict[str, str]] = []
        for message in chat.messages:
            if message.role == "system":
                system.append({"text": message.content})
                continue
            role = "model" if message.role == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": message.content}]})

        body: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": chat.temperature,
                "maxOutputTokens": chat.max_output_tokens,
                "responseMimeType": "application/json",
                "responseSchema": _strip_schema(chat.json_schema),
            },
        }
        if system:
            body["systemInstruction"] = {"parts": system}
        return body

    def _parse(self, payload: dict[str, Any]) -> RawCompletion:
        usage_block = payload.get("usageMetadata")
        usage = (
            usage_from(
                usage_block,
                prompt="promptTokenCount",
                completion="candidatesTokenCount",
                total="totalTokenCount",
            )
            if isinstance(usage_block, dict)
            else TokenUsage()
        )

        feedback = payload.get("promptFeedback") or {}
        blocked = feedback.get("blockReason")
        if isinstance(blocked, str) and blocked:
            return RawCompletion(
                usage=usage,
                refusal=Refusal(
                    reason=f"gemini blocked the prompt: {blocked}", provider_code=blocked
                ),
                raw=payload,
            )

        candidates = payload.get("candidates") or []
        if not candidates:
            message = f"gemini returned no candidates: {str(payload)[:200]}"
            raise ProviderError(message)

        candidate = candidates[0]
        reason = candidate.get("finishReason")
        text = "".join(
            part.get("text", "")
            for part in (candidate.get("content") or {}).get("parts") or []
            if isinstance(part, dict)
        )

        if isinstance(reason, str) and reason in REFUSAL_REASONS:
            return RawCompletion(
                usage=usage,
                refusal=Refusal(
                    reason=f"gemini stopped generating: {reason}", provider_code=reason
                ),
                raw=payload,
            )

        if not text.strip():
            # MAX_TOKENS with no text is a truncation, not a refusal: the model
            # was willing and ran out of room, and the next provider may not.
            message = f"gemini returned no text (finishReason {reason!r}): {str(payload)[:200]}"
            raise ProviderError(message)

        return RawCompletion(text=text, usage=usage, raw=payload)
