"""The judge is never the model it grades, and the collision is a skip.

The two cases that matter are the two the correction to the plan called out: a
collision mid-chain must be stepped over so the call still succeeds, and a chain
with only one usable model must fail with the reason named rather than with a
generic exhaustion.
"""

from __future__ import annotations

import json

import pytest
from fakes import FakeProvider, Verdict

from signaldesk.core.errors import RateLimitError
from signaldesk.rag.llm.base import Message
from signaldesk.rag.llm.errors import AllProvidersFailedError, SameModelError
from signaldesk.rag.llm.judge import judge as judge_call
from signaldesk.rag.llm.types import AttemptOutcome, ModelIdentity

pytestmark = pytest.mark.unit

GOOD = json.dumps({"label": "agree", "confident": True})


def _judge(chain: list[FakeProvider], messages: list[Message], generated_by: ModelIdentity):
    return judge_call(
        messages,
        schema=Verdict,
        generated_by=generated_by,
        prompt_name="judge",
        prompt_version="judge_v1",
        chain=chain,  # type: ignore[arg-type]
    )


def test_a_collision_is_skipped_and_the_next_provider_answers(
    messages: list[Message],
) -> None:
    """The failover case: the generator fell through to groq, the judge is groq.

    Raising here would kill the call with cerebras sitting unused.
    """
    chain = [FakeProvider("groq", text=GOOD), FakeProvider("cerebras", text=GOOD)]
    result = _judge(chain, messages, ModelIdentity(provider="groq", model="m"))

    assert result.run.identity == ModelIdentity(provider="cerebras", model="m")
    assert result.trace.attempts[0].outcome is AttemptOutcome.SKIPPED_SAME_MODEL
    assert not chain[0].calls


def test_only_the_colliding_model_is_skipped(messages: list[Message]) -> None:
    """A different model at the same provider is a different model."""
    chain = [FakeProvider("groq", model="a", text=GOOD)]
    result = _judge(chain, messages, ModelIdentity(provider="groq", model="b"))
    assert result.run.identity == ModelIdentity(provider="groq", model="a")


def test_a_chain_with_only_one_usable_model_raises(messages: list[Message]) -> None:
    chain = [FakeProvider("groq", text=GOOD)]
    with pytest.raises(SameModelError) as caught:
        _judge(chain, messages, ModelIdentity(provider="groq", model="m"))
    message = str(caught.value)
    assert "groq/m" in message
    assert "never be the model it is grading" in message


def test_the_error_lists_what_collided(messages: list[Message]) -> None:
    chain = [
        FakeProvider("groq", text=GOOD),
        FakeProvider("cerebras", configured=False),
    ]
    with pytest.raises(SameModelError) as caught:
        _judge(chain, messages, ModelIdentity(provider="groq", model="m"))
    assert len(caught.value.trace.collided) == 1


def test_real_failures_beat_the_collision_in_the_error(messages: list[Message]) -> None:
    """If two providers were rate limited and one collided, the story is the
    rate limiting. SameModelError would send the reader to the wrong fix."""
    chain = [
        FakeProvider("groq", text=GOOD),
        FakeProvider("cerebras", raises=RateLimitError("429")),
    ]
    with pytest.raises(AllProvidersFailedError) as caught:
        _judge(chain, messages, ModelIdentity(provider="groq", model="m"))
    assert not isinstance(caught.value, SameModelError)
    assert "rate_limit" in str(caught.value)


def test_a_judge_that_never_collides_is_unaffected(messages: list[Message]) -> None:
    chain = [FakeProvider("groq", text=GOOD)]
    result = _judge(chain, messages, ModelIdentity(provider="gemini", model="flash"))
    assert result.run.identity.provider == "groq"
    assert not result.trace.collided
