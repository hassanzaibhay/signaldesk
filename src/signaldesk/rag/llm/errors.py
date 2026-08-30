"""Failures this layer raises.

All four root in ``ProviderError`` from ``core/errors.py``, so a caller that
catches the core type catches these. They are declared here rather than added to
the core module for the same reason ``ContingencyError``,
``ReferenceSetError`` and ``SignalScopeError`` are declared beside the code that
raises them: the error and the invariant it defends should be readable in one
place.
"""

from __future__ import annotations

from signaldesk.core.errors import ProviderError
from signaldesk.rag.llm.types import CallTrace


class AllProvidersFailedError(ProviderError):
    """Every provider in the chain was skipped or failed.

    Carries the trace, so the caller can see one line per provider rather than
    only the last thing that went wrong. ``nothing_configured`` separates the
    deployment mistake - no keys anywhere - from four providers that were asked
    and could not answer.
    """

    def __init__(self, trace: CallTrace, *, what: str = "completion") -> None:
        self.trace = trace
        self.nothing_configured = trace.nothing_configured
        if trace.nothing_configured:
            message = (
                f"no provider was available for this {what}: {trace.render()}. "
                "Set GEMINI_API_KEY, GROQ_API_KEY or CEREBRAS_API_KEY, or set "
                "OLLAMA_ENABLED=true with a model pulled locally."
            )
        else:
            message = f"every provider failed for this {what}: {trace.render()}"
        super().__init__(message)


class SameModelError(ProviderError):
    """No judge remained that differs from the model being judged.

    A judge that is the same model as the generator is grading its own work, so
    a collision means that provider is skipped. This is raised only when the
    whole chain has been walked and every usable provider collided - the chain
    is exhausted and no distinct model remained.
    """

    def __init__(self, generated_by: object, trace: CallTrace) -> None:
        self.trace = trace
        self.generated_by = generated_by
        collided = ", ".join(str(attempt.identity) for attempt in trace.collided)
        message = (
            f"no judge distinct from {generated_by} was available; the judge must "
            f"never be the model it is grading. Skipped as identical: {collided or 'none'}. "
            f"Full chain: {trace.render()}"
        )
        super().__init__(message)


class CassetteMissError(ProviderError):
    """Replay was asked for a cassette that is not on disk.

    Raised rather than falling through to a live call. A test suite that
    silently reached the network when a cassette was missing would pass on a
    machine with keys and fail in CI, which is the failure this mode exists to
    prevent.
    """

    def __init__(self, key: str, path: object, *, detail: str = "") -> None:
        self.key = key
        self.path = path
        suffix = f" {detail}" if detail else ""
        message = (
            f"no cassette for key {key}; expected {path}. Record it with "
            f"'signaldesk evals record-cassettes' - replay never falls back to a "
            f"live call.{suffix}"
        )
        super().__init__(message)


class ProviderNotAllowedError(ProviderError):
    """A provider was configured to point somewhere that is not free.

    Raised at construction, before any request. Total infrastructure cost is
    zero, and the way that stays true under configuration drift is that a base
    URL or model outside the allowlist cannot be built into a chain at all.
    """

    def __init__(self, provider: str, *, endpoint: str = "", model: str = "") -> None:
        self.provider = provider
        detail = []
        if endpoint:
            detail.append(f"endpoint {endpoint!r}")
        if model:
            detail.append(f"model {model!r}")
        message = (
            f"{provider} is configured with {' and '.join(detail) or 'unknown settings'}, "
            "which is not on the free-tier allowlist. This project runs at zero "
            "infrastructure cost and no paid endpoint may be reachable from config."
        )
        super().__init__(message)
