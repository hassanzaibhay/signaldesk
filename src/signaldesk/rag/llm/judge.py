"""Judging, with the guarantee that the judge is never the generator.

A judge that is the same model as the thing it is grading is marking its own
work, and the resulting agreement number measures nothing. So the identity of
the generation is passed in and the router steps over any provider that would
resolve to it.

## Why this skips rather than raises

The collision is not a configuration mistake, it is a normal consequence of
failover. The generator is configured Gemini and falls through to Groq; the
judge is configured Groq; they are now the same model through nobody's error.
Raising there would turn a routine failover into a dead call with Cerebras still
sitting unused in the chain.

So a collision is a skip, recorded as ``skipped_same_model``, and the walk
continues. ``SameModelError`` is raised only when the chain is exhausted and no
distinct model remained - the guarantee is identical, and the failure mode is
gone.

An exhausted chain that also contains real failures raises
``AllProvidersFailedError`` instead: if two providers were rate limited and one
collided, the story is the rate limiting, not the collision.
"""

from __future__ import annotations

from collections.abc import Sequence

from signaldesk.core.config import Settings, get_settings
from signaldesk.core.logging import get_logger
from signaldesk.rag.llm import registry, router
from signaldesk.rag.llm.base import Message, Provider
from signaldesk.rag.llm.errors import AllProvidersFailedError, SameModelError
from signaldesk.rag.llm.types import Completion, ModelIdentity, Schema

log = get_logger(__name__)


def judge(
    messages: Sequence[Message],
    *,
    schema: type[Schema],
    generated_by: ModelIdentity,
    prompt_name: str,
    prompt_version: str,
    chain: Sequence[Provider] | None = None,
    settings: Settings | None = None,
    max_output_tokens: int = 2048,
) -> Completion[Schema]:
    """Grade a generation with a model that is not the one that produced it.

    ``generated_by`` is the identity from the generation's ``ModelRun``, so the
    comparison is against the model that actually answered rather than the one
    that was configured to.
    """
    settings = settings or get_settings()
    providers = tuple(chain) if chain is not None else registry.judge_chain(settings)

    try:
        result = router.complete(
            messages,
            schema=schema,
            prompt_name=prompt_name,
            prompt_version=prompt_version,
            chain=providers,
            exclude=(generated_by,),
            settings=settings,
            max_output_tokens=max_output_tokens,
            what="judgement",
        )
    except AllProvidersFailedError as error:
        # Every provider was skipped or failed. If the only thing standing
        # between this call and an answer was the same-model rule, say that -
        # it is a different problem with a different fix.
        if error.trace.collided and not error.trace.tried:
            raise SameModelError(generated_by, error.trace) from error
        raise

    log.info(
        "llm.judge.answered",
        judge=str(result.run.identity),
        generated_by=str(generated_by),
        skipped_same_model=len(result.trace.collided),
    )
    return result
