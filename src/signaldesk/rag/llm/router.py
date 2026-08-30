"""The failover loop. One call in, one typed answer or one typed error out.

## What fails over, and what does not

Down the chain: transport failure, rate limiting, an unusable HTTP status, and
output that will not validate against the caller's schema after one repair. All
four mean this provider did not answer.

Not down the chain: a refusal. A model that declines has told the caller
something, and walking to the next provider to shop for a more compliant one
would be both wasteful and wrong. A refusal is returned as a result.

## What the caller sees when everything fails

``AllProvidersFailedError``, carrying the trace. Never ``None``, never a partial
value, never the last provider's raw text. The trace holds one entry per
provider with the reason it produced nothing, and its ``nothing_configured``
flag separates "no keys anywhere" - a deployment mistake - from "four providers
were asked and all four failed".

## Retries

There is no retry loop here. Retrying inside a provider is
``core/http.py``'s tenacity policy, which the provider call already goes
through: three attempts, exponential backoff with jitter, ``Retry-After``
honoured. The only loop in this module is across providers, and it starts where
that policy has already given up. One retry philosophy, and it lives in
``core/http.py``.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable, Sequence

import httpx

from signaldesk.core.config import Settings, get_settings
from signaldesk.core.errors import ProviderError, RateLimitError, StructuredOutputError
from signaldesk.core.logging import get_logger
from signaldesk.rag.llm import registry, structured
from signaldesk.rag.llm.base import ChatRequest, Message, Provider, RawCompletion
from signaldesk.rag.llm.errors import AllProvidersFailedError, CassetteMissError
from signaldesk.rag.llm.types import (
    Attempt,
    AttemptOutcome,
    CallTrace,
    Completion,
    ModelIdentity,
    ModelRun,
    Schema,
    TokenUsage,
)

log = get_logger(__name__)


def prompt_digest(messages: Sequence[Message]) -> str:
    """A hash of the exact rendered input, for the provenance record."""
    canonical = json.dumps(
        [{"role": m.role, "content": m.content} for m in messages], sort_keys=True
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


def _attempt_provider(
    provider: Provider,
    chat: ChatRequest,
    schema: type[Schema],
) -> tuple[RawCompletion | None, Schema | None, AttemptOutcome, str, int]:
    """Ask one provider. Returns (raw, value, outcome, detail, repairs).

    Everything provider-shaped is caught and classified here so the loop below
    reads as a walk rather than as a pile of exception handling. Cassette
    resolution is the provider's business, not this function's.
    """
    try:
        raw = provider.complete(chat)
    except CassetteMissError:
        # Not a provider failure. Replay is missing a recording, which is a
        # problem with the test corpus rather than with this provider, and
        # walking down the chain would bury it as "everything failed".
        raise
    except RateLimitError as error:
        return None, None, AttemptOutcome.RATE_LIMIT, str(error)[:300], 0
    except httpx.TransportError as error:
        return None, None, AttemptOutcome.TRANSPORT, repr(error)[:300], 0
    except ProviderError as error:
        return None, None, AttemptOutcome.HTTP_STATUS, str(error)[:300], 0

    if raw.refusal is not None:
        return raw, None, AttemptOutcome.REFUSED, raw.refusal.reason[:300], 0

    try:
        value = structured.parse(raw.text, schema)
    except StructuredOutputError as first:
        problem = str(first)
    else:
        return raw, value, AttemptOutcome.OK, "", 0  # type: ignore[return-value]

    # One repair, against the same provider, handing back its own mistakes.
    repair = ChatRequest(
        messages=structured.repair_messages(chat.messages, raw.text, problem, schema),
        json_schema=chat.json_schema,
        schema_name=chat.schema_name,
        prompt_version=chat.prompt_version,
        temperature=chat.temperature,
        max_output_tokens=chat.max_output_tokens,
    )
    try:
        repaired = provider.complete(repair)
    except CassetteMissError:
        raise
    except (RateLimitError, httpx.TransportError, ProviderError) as error:
        return None, None, AttemptOutcome.SCHEMA, f"repair attempt failed: {error!r}"[:300], 1

    if repaired.refusal is not None:
        return repaired, None, AttemptOutcome.REFUSED, repaired.refusal.reason[:300], 1

    try:
        value = structured.parse(repaired.text, schema)
    except StructuredOutputError as second:
        detail = f"{problem} | after repair: {second}"[:300]
        return None, None, AttemptOutcome.SCHEMA, detail, 1

    return repaired, value, AttemptOutcome.OK, "", 1  # type: ignore[return-value]


def complete(
    messages: Sequence[Message],
    *,
    schema: type[Schema],
    prompt_name: str,
    prompt_version: str,
    chain: Sequence[Provider] | None = None,
    exclude: Iterable[ModelIdentity] = (),
    settings: Settings | None = None,
    temperature: float | None = None,
    max_output_tokens: int = 2048,
    what: str = "completion",
) -> Completion[Schema]:
    """Walk the chain until one provider answers, or raise.

    ``exclude`` names identities to skip without asking. The judge uses it to
    step over the model it is grading; skipping is recorded in the trace as
    ``skipped_same_model`` rather than being silent.
    """
    settings = settings or get_settings()
    providers = tuple(chain) if chain is not None else registry.generation_chain(settings)
    excluded = frozenset(exclude)

    chat = ChatRequest(
        messages=tuple(messages),
        json_schema=structured.json_schema_for(schema),
        schema_name=schema.__name__,
        prompt_version=prompt_version,
        temperature=0.0 if temperature is None else temperature,
        max_output_tokens=max_output_tokens,
    )
    digest = prompt_digest(chat.messages)
    attempts: list[Attempt] = []

    for provider in providers:
        identity = provider.identity

        if identity in excluded:
            attempts.append(Attempt(identity=identity, outcome=AttemptOutcome.SKIPPED_SAME_MODEL))
            continue
        if not provider.is_configured:
            attempts.append(Attempt(identity=identity, outcome=AttemptOutcome.SKIPPED_UNCONFIGURED))
            continue

        started = time.monotonic()
        raw, value, outcome, detail, repairs = _attempt_provider(provider, chat, schema)
        latency_ms = (time.monotonic() - started) * 1000.0
        attempts.append(
            Attempt(
                identity=identity,
                outcome=outcome,
                detail=detail,
                latency_ms=round(latency_ms, 2),
                repairs=repairs,
            )
        )

        if outcome not in {AttemptOutcome.OK, AttemptOutcome.REFUSED}:
            log.warning(
                "llm.provider.failed",
                provider=identity.provider,
                model=identity.model,
                outcome=str(outcome),
                detail=detail,
            )
            continue

        trace = CallTrace(attempts=tuple(attempts))
        run = ModelRun(
            identity=identity,
            prompt_name=prompt_name,
            prompt_version=prompt_version,
            prompt_digest=digest,
            usage=raw.usage if raw is not None else TokenUsage(),
            latency_ms=round(latency_ms, 2),
            cassette_key=raw.cassette_key if raw is not None else None,
        )
        log.info(
            "llm.call.answered",
            provider=identity.provider,
            model=identity.model,
            prompt_version=prompt_version,
            outcome=str(outcome),
            latency_ms=run.latency_ms,
            skipped=len(trace.attempts) - 1,
        )
        return Completion(
            value=value,
            refusal=raw.refusal if raw else None,
            run=run,
            trace=trace,
        )

    trace = CallTrace(attempts=tuple(attempts))
    log.error("llm.chain.exhausted", trace=trace.render(), what=what)
    raise AllProvidersFailedError(trace, what=what)
