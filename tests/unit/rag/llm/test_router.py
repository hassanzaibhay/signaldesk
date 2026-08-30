"""The failover loop: what moves down the chain, and what does not."""

from __future__ import annotations

import json

import httpx
import pytest
from fakes import FakeProvider, Verdict

from signaldesk.core.errors import ProviderError, RateLimitError
from signaldesk.rag.llm import router
from signaldesk.rag.llm.base import Message
from signaldesk.rag.llm.errors import AllProvidersFailedError
from signaldesk.rag.llm.types import AttemptOutcome, ModelIdentity, Refusal

pytestmark = pytest.mark.unit

GOOD = json.dumps({"label": "labelled", "confident": True})
OTHER = json.dumps({"label": "unlabelled", "confident": False})


def _complete(chain: list[FakeProvider], messages: list[Message], **kwargs: object):
    return router.complete(
        messages,
        schema=Verdict,
        prompt_name="probe",
        prompt_version="probe_v1",
        chain=chain,  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )


class TestTheHappyPath:
    def test_the_first_configured_provider_answers(self, messages: list[Message]) -> None:
        chain = [FakeProvider("gemini", text=GOOD), FakeProvider("groq", text=OTHER)]
        result = _complete(chain, messages)
        assert result.value == Verdict(label="labelled", confident=True)
        assert result.run.identity == ModelIdentity(provider="gemini", model="m")
        assert not chain[1].calls

    def test_the_provenance_record_is_populated(self, messages: list[Message]) -> None:
        result = _complete([FakeProvider("gemini", text=GOOD)], messages)
        assert result.run.prompt_name == "probe"
        assert result.run.prompt_version == "probe_v1"
        assert result.run.prompt_digest
        assert result.run.usage.total == 7
        assert result.run.latency_ms >= 0.0

    def test_the_digest_changes_with_the_input(self, messages: list[Message]) -> None:
        """Provenance has to distinguish two calls to one prompt version."""
        first = _complete([FakeProvider("gemini", text=GOOD)], messages)
        second = _complete(
            [FakeProvider("gemini", text=GOOD)], [Message(role="user", content="different")]
        )
        assert first.run.prompt_digest != second.run.prompt_digest


class TestWhatFailsOver:
    @pytest.mark.parametrize(
        ("error", "outcome"),
        [
            (RateLimitError("429"), AttemptOutcome.RATE_LIMIT),
            (httpx.ConnectError("refused"), AttemptOutcome.TRANSPORT),
            (ProviderError("500"), AttemptOutcome.HTTP_STATUS),
        ],
    )
    def test_transport_rate_limit_and_bad_status_move_down_the_chain(
        self, messages: list[Message], error: Exception, outcome: AttemptOutcome
    ) -> None:
        chain = [FakeProvider("gemini", raises=error), FakeProvider("groq", text=GOOD)]
        result = _complete(chain, messages)
        assert result.run.identity.provider == "groq"
        assert result.trace.attempts[0].outcome is outcome

    def test_output_that_will_not_validate_moves_down_the_chain(
        self, messages: list[Message]
    ) -> None:
        """Two bad responses: the first, then the repair. Then the next provider."""
        chain = [
            FakeProvider("gemini", texts=["not json", "still not json"]),
            FakeProvider("groq", text=GOOD),
        ]
        result = _complete(chain, messages)
        assert result.run.identity.provider == "groq"
        assert result.trace.attempts[0].outcome is AttemptOutcome.SCHEMA
        assert result.trace.attempts[0].repairs == 1

    def test_the_whole_chain_is_walked_in_order(self, messages: list[Message]) -> None:
        chain = [
            FakeProvider("gemini", raises=RateLimitError("429")),
            FakeProvider("groq", raises=ProviderError("500")),
            FakeProvider("cerebras", text=GOOD),
        ]
        result = _complete(chain, messages)
        assert [a.identity.provider for a in result.trace.attempts] == [
            "gemini",
            "groq",
            "cerebras",
        ]


class TestOneRepairThenGiveUp:
    def test_a_repaired_response_is_accepted(self, messages: list[Message]) -> None:
        provider = FakeProvider("gemini", texts=["not json", GOOD])
        result = _complete([provider], messages)
        assert result.value == Verdict(label="labelled", confident=True)
        assert result.trace.attempts[0].repairs == 1
        assert len(provider.calls) == 2

    def test_the_repair_turn_carries_the_rejected_text_and_the_schema(
        self, messages: list[Message]
    ) -> None:
        provider = FakeProvider("gemini", texts=["not json", GOOD])
        _complete([provider], messages)
        repair = provider.calls[1]
        assert repair.messages[-2].role == "assistant"
        assert repair.messages[-2].content == "not json"
        assert "schema" in repair.messages[-1].content.lower()

    def test_there_is_no_second_repair(self, messages: list[Message]) -> None:
        """Two attempts inside a provider, never three."""
        provider = FakeProvider("gemini", texts=["bad", "still bad", GOOD])
        with pytest.raises(AllProvidersFailedError):
            _complete([provider], messages)
        assert len(provider.calls) == 2


class TestRefusalIsAResultNotAFailure:
    def test_a_refusal_stops_the_walk(self, messages: list[Message]) -> None:
        """Shopping down the chain for a more compliant model would be wrong."""
        chain = [
            FakeProvider("gemini", refusal=Refusal(reason="declined", provider_code="SAFETY")),
            FakeProvider("groq", text=GOOD),
        ]
        result = _complete(chain, messages)
        assert result.refused
        assert result.value is None
        assert result.refusal is not None
        assert result.refusal.reason == "declined"
        assert not chain[1].calls

    def test_a_refusal_still_carries_provenance(self, messages: list[Message]) -> None:
        chain = [FakeProvider("gemini", refusal=Refusal(reason="declined"))]
        result = _complete(chain, messages)
        assert result.run.identity.provider == "gemini"
        assert result.trace.attempts[0].outcome is AttemptOutcome.REFUSED

    def test_unwrap_raises_on_a_refusal(self, messages: list[Message]) -> None:
        chain = [FakeProvider("gemini", refusal=Refusal(reason="declined"))]
        with pytest.raises(ValueError, match="refused"):
            _complete(chain, messages).unwrap()


class TestUnconfiguredProvidersAreSkipped:
    def test_a_provider_with_no_key_is_skipped_not_failed(self, messages: list[Message]) -> None:
        chain = [FakeProvider("gemini", configured=False), FakeProvider("groq", text=GOOD)]
        result = _complete(chain, messages)
        assert result.trace.attempts[0].outcome is AttemptOutcome.SKIPPED_UNCONFIGURED
        assert not chain[0].calls

    def test_an_unconfigured_provider_is_never_asked(self, messages: list[Message]) -> None:
        provider = FakeProvider("gemini", configured=False, raises=ProviderError("must not"))
        chain = [provider, FakeProvider("groq", text=GOOD)]
        result = _complete(chain, messages)
        assert result.run.identity.provider == "groq"


class TestWhenEveryProviderFails:
    def test_it_raises_rather_than_returning_anything(self, messages: list[Message]) -> None:
        chain = [
            FakeProvider("gemini", raises=RateLimitError("429")),
            FakeProvider("groq", raises=httpx.ConnectError("refused")),
        ]
        with pytest.raises(AllProvidersFailedError) as caught:
            _complete(chain, messages)
        assert not caught.value.nothing_configured

    def test_the_error_names_every_provider_and_its_reason(self, messages: list[Message]) -> None:
        chain = [
            FakeProvider("gemini", raises=RateLimitError("429 slow down")),
            FakeProvider("groq", raises=ProviderError("500 upstream")),
        ]
        with pytest.raises(AllProvidersFailedError) as caught:
            _complete(chain, messages)
        message = str(caught.value)
        assert "gemini" in message
        assert "groq" in message
        assert "rate_limit" in message
        assert "http_status" in message

    def test_nothing_configured_is_distinguishable_from_everything_failing(
        self, messages: list[Message]
    ) -> None:
        """A deployment mistake reads differently from four providers failing."""
        chain = [
            FakeProvider("gemini", configured=False),
            FakeProvider("groq", configured=False),
        ]
        with pytest.raises(AllProvidersFailedError) as caught:
            _complete(chain, messages)
        assert caught.value.nothing_configured
        assert "GEMINI_API_KEY" in str(caught.value)

    def test_an_empty_chain_raises_rather_than_returning_none(
        self, messages: list[Message]
    ) -> None:
        with pytest.raises(AllProvidersFailedError):
            _complete([], messages)

    def test_the_trace_survives_on_the_exception(self, messages: list[Message]) -> None:
        chain = [FakeProvider("gemini", raises=ProviderError("500"))]
        with pytest.raises(AllProvidersFailedError) as caught:
            _complete(chain, messages)
        assert len(caught.value.trace.attempts) == 1
        assert caught.value.trace.tried


class TestExclusion:
    def test_an_excluded_identity_is_skipped_and_recorded(self, messages: list[Message]) -> None:
        chain = [FakeProvider("groq", text=GOOD), FakeProvider("cerebras", text=OTHER)]
        result = _complete(chain, messages, exclude=(ModelIdentity(provider="groq", model="m"),))
        assert result.run.identity.provider == "cerebras"
        assert result.trace.attempts[0].outcome is AttemptOutcome.SKIPPED_SAME_MODEL
        assert not chain[0].calls
