"""Each provider against its own JSON, replayed through the real transport.

These run the provider's actual request building and response parsing against
the committed cassettes. That is the half of a provider module a higher-level
stub would skip, and it is where the shape assumptions live.

Every body here is constructed rather than captured, so what these tests prove
is that the parsing matches the documentation, not that the documentation
matches the API.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from fakes import Verdict

from signaldesk.core.config import Settings
from signaldesk.core.errors import ProviderError, RateLimitError
from signaldesk.rag.llm import cassettes, structured
from signaldesk.rag.llm.base import ChatRequest, Message
from signaldesk.rag.llm.providers.cerebras import CerebrasProvider
from signaldesk.rag.llm.providers.gemini import GeminiProvider
from signaldesk.rag.llm.providers.groq import GroqProvider
from signaldesk.rag.llm.providers.ollama import OllamaProvider, host_is_local

pytestmark = pytest.mark.unit

ANSWER = '{"label": "labelled", "confident": true}'


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    fields: dict[str, object] = {
        "django_secret_key": "test",
        "data_dir": tmp_path,
        "cache_dir": tmp_path / "cache",
        "gemini_api_key": "g",
        "groq_api_key": "q",
        "cerebras_api_key": "c",
        "ollama_enabled": True,
    }
    fields.update(overrides)
    return Settings(**fields)  # type: ignore[arg-type]


def _chat() -> ChatRequest:
    return ChatRequest(
        messages=(
            Message(role="system", content="Be exact."),
            Message(role="user", content="Is nausea labelled?"),
        ),
        json_schema=structured.json_schema_for(Verdict),
        schema_name="Verdict",
        prompt_version="v1",
    )


def _transport(body: dict[str, object], status: int = 200) -> httpx.MockTransport:
    return cassettes.replay_transport(
        cassettes.Cassette(
            key="k",
            provider="p",
            model="m",
            prompt_version="v1",
            schema_name="Verdict",
            status_code=status,
            body=body,
            constructed=True,
        )
    )


def _openai_body(content: str = ANSWER, **extra: object) -> dict[str, object]:
    choice: dict[str, object] = {
        "index": 0,
        "message": {"role": "assistant", "content": content},
        "finish_reason": "stop",
    }
    choice.update(extra)
    return {
        "choices": [choice],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
    }


class TestChatCompletionsProviders:
    @pytest.mark.parametrize("cls", [GroqProvider, CerebrasProvider])
    def test_the_answer_and_usage_are_parsed(self, tmp_path: Path, cls: type) -> None:
        provider = cls("m", _settings(tmp_path))
        raw = provider.complete(_chat(), transport=_transport(_openai_body()))
        assert raw.text == ANSWER
        assert raw.usage.prompt == 10
        assert raw.usage.total == 14
        assert raw.refusal is None

    def test_the_request_carries_the_schema_and_the_key(self, tmp_path: Path) -> None:
        seen: dict[str, object] = {}

        def _handler(request: httpx.Request) -> httpx.Response:
            seen["auth"] = request.headers.get("Authorization")
            seen["body"] = request.read().decode()
            return httpx.Response(200, json=_openai_body())

        provider = GroqProvider("llama-3.3-70b-versatile", _settings(tmp_path))
        provider.complete(_chat(), transport=httpx.MockTransport(_handler))
        assert seen["auth"] == "Bearer q"
        assert "json_schema" in str(seen["body"])
        assert "llama-3.3-70b-versatile" in str(seen["body"])

    def test_an_explicit_refusal_is_a_refusal_not_a_failure(self, tmp_path: Path) -> None:
        body = {
            "choices": [{"index": 0, "message": {"role": "assistant", "refusal": "I cannot help."}}]
        }
        provider = GroqProvider("m", _settings(tmp_path))
        raw = provider.complete(_chat(), transport=_transport(body))
        assert raw.refusal is not None
        assert raw.refusal.reason == "I cannot help."

    def test_a_content_filter_stop_is_a_refusal(self, tmp_path: Path) -> None:
        body = _openai_body(content="", finish_reason="content_filter")
        provider = GroqProvider("m", _settings(tmp_path))
        raw = provider.complete(_chat(), transport=_transport(body))
        assert raw.refusal is not None
        assert raw.refusal.provider_code == "content_filter"

    def test_a_429_is_a_rate_limit(self, tmp_path: Path) -> None:
        provider = GroqProvider("m", _settings(tmp_path))
        with pytest.raises(RateLimitError):
            provider.complete(_chat(), transport=_transport({"error": "slow"}, status=429))

    def test_another_bad_status_is_a_provider_error(self, tmp_path: Path) -> None:
        provider = GroqProvider("m", _settings(tmp_path))
        with pytest.raises(ProviderError, match="400"):
            provider.complete(_chat(), transport=_transport({"error": "bad"}, status=400))

    def test_no_choices_is_a_provider_error(self, tmp_path: Path) -> None:
        provider = GroqProvider("m", _settings(tmp_path))
        with pytest.raises(ProviderError, match="no choices"):
            provider.complete(_chat(), transport=_transport({"choices": []}))


class TestGemini:
    def test_the_answer_and_usage_are_parsed(self, tmp_path: Path) -> None:
        body = {
            "candidates": [
                {"content": {"parts": [{"text": ANSWER}], "role": "model"}, "finishReason": "STOP"}
            ],
            "usageMetadata": {
                "promptTokenCount": 11,
                "candidatesTokenCount": 5,
                "totalTokenCount": 16,
            },
        }
        provider = GeminiProvider("gemini-2.5-flash", _settings(tmp_path))
        raw = provider.complete(_chat(), transport=_transport(body))
        assert raw.text == ANSWER
        assert raw.usage.total == 16

    def test_multi_part_text_is_joined(self, tmp_path: Path) -> None:
        body = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": '{"label": "a", '}, {"text": '"confident": true}'}]
                    },
                    "finishReason": "STOP",
                }
            ]
        }
        provider = GeminiProvider("gemini-2.5-flash", _settings(tmp_path))
        assert provider.complete(_chat(), transport=_transport(body)).text == (
            '{"label": "a", "confident": true}'
        )

    def test_a_blocked_prompt_is_a_refusal(self, tmp_path: Path) -> None:
        body = {"promptFeedback": {"blockReason": "SAFETY"}}
        provider = GeminiProvider("gemini-2.5-flash", _settings(tmp_path))
        raw = provider.complete(_chat(), transport=_transport(body))
        assert raw.refusal is not None
        assert raw.refusal.provider_code == "SAFETY"

    def test_a_safety_finish_reason_is_a_refusal(self, tmp_path: Path) -> None:
        body = {"candidates": [{"content": {"parts": []}, "finishReason": "SAFETY"}]}
        provider = GeminiProvider("gemini-2.5-flash", _settings(tmp_path))
        raw = provider.complete(_chat(), transport=_transport(body))
        assert raw.refusal is not None

    def test_a_truncation_is_a_failure_not_a_refusal(self, tmp_path: Path) -> None:
        """The model was willing and ran out of room. The next provider may not be."""
        body = {"candidates": [{"content": {"parts": []}, "finishReason": "MAX_TOKENS"}]}
        provider = GeminiProvider("gemini-2.5-flash", _settings(tmp_path))
        with pytest.raises(ProviderError, match="no text"):
            provider.complete(_chat(), transport=_transport(body))

    def test_the_system_turn_becomes_a_system_instruction(self, tmp_path: Path) -> None:
        seen: dict[str, object] = {}

        def _handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = request.read().decode()
            seen["key"] = request.headers.get("x-goog-api-key")
            return httpx.Response(
                200,
                json={"candidates": [{"content": {"parts": [{"text": ANSWER}]}}]},
            )

        provider = GeminiProvider("gemini-2.5-flash", _settings(tmp_path))
        provider.complete(_chat(), transport=httpx.MockTransport(_handler))
        assert "systemInstruction" in str(seen["body"])
        assert seen["key"] == "g"

    def test_the_key_is_not_in_the_url(self, tmp_path: Path) -> None:
        """Gemini accepts the key as a query parameter too. A URL gets logged."""
        secret = "sd-test-key-not-a-real-credential"
        seen: dict[str, object] = {}

        def _handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["header"] = request.headers.get("x-goog-api-key")
            return httpx.Response(
                200, json={"candidates": [{"content": {"parts": [{"text": ANSWER}]}}]}
            )

        GeminiProvider("gemini-2.5-flash", _settings(tmp_path, gemini_api_key=secret)).complete(
            _chat(), transport=httpx.MockTransport(_handler)
        )
        assert secret not in str(seen["url"])
        assert seen["header"] == secret

    def test_unsupported_schema_keywords_are_stripped(self, tmp_path: Path) -> None:
        provider = GeminiProvider("gemini-2.5-flash", _settings(tmp_path))
        body = provider._body(_chat())
        rendered = str(body["generationConfig"])
        assert "$defs" not in rendered
        assert "additionalProperties" not in rendered


class TestOllama:
    def test_the_answer_and_token_counts_are_parsed(self, tmp_path: Path) -> None:
        body = {
            "message": {"role": "assistant", "content": ANSWER},
            "done": True,
            "prompt_eval_count": 12,
            "eval_count": 6,
        }
        provider = OllamaProvider("qwen2.5:7b-instruct", _settings(tmp_path))
        raw = provider.complete(_chat(), transport=_transport(body))
        assert raw.text == ANSWER
        assert raw.usage.prompt == 12
        assert raw.usage.total == 18

    def test_missing_counts_stay_none_rather_than_zero(self, tmp_path: Path) -> None:
        """A zero would be a fabricated measurement once these are summed."""
        body = {"message": {"content": ANSWER}, "done": True}
        provider = OllamaProvider("m", _settings(tmp_path))
        raw = provider.complete(_chat(), transport=_transport(body))
        assert raw.usage.prompt is None
        assert raw.usage.total is None

    def test_the_schema_is_sent_as_the_format(self, tmp_path: Path) -> None:
        provider = OllamaProvider("m", _settings(tmp_path))
        assert provider._body(_chat())["format"] == structured.json_schema_for(Verdict)

    def test_an_empty_message_is_a_provider_error(self, tmp_path: Path) -> None:
        provider = OllamaProvider("m", _settings(tmp_path))
        with pytest.raises(ProviderError, match="empty message"):
            provider.complete(_chat(), transport=_transport({"message": {"content": ""}}))

    @pytest.mark.parametrize(
        ("url", "local"),
        [
            ("http://localhost:11434", True),
            ("http://ollama:11434", True),
            ("https://api.some-host.com", False),
            ("not a url at all", False),
        ],
    )
    def test_locality_of_the_base_url(self, url: str, local: bool) -> None:
        assert host_is_local(url) is local
