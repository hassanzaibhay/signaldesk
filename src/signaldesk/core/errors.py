"""Typed error hierarchy.

Every failure raised by this project is one of these. Nothing raises a bare
``Exception``, and nothing returns a silent default in place of an error: in a
statistics pipeline a silent default is a fabricated result.
"""

from __future__ import annotations


class SignalDeskError(Exception):
    """Base class for every error this project raises."""


class IngestError(SignalDeskError):
    """A data ingest step failed."""


class SchemaMismatchError(IngestError):
    """An input file's columns do not match the schema declared for its era."""


class NormalizationError(IngestError):
    """A source term could not be normalized to a controlled vocabulary."""


class EstimatorConvergenceError(SignalDeskError):
    """A statistical estimator failed to converge; no value is returned."""


class RetrievalError(SignalDeskError):
    """The retrieval pipeline could not produce candidates."""


class ProviderError(SignalDeskError):
    """A model provider call failed."""


class RateLimitError(ProviderError):
    """A model provider rejected the call with a rate limit."""


class StructuredOutputError(ProviderError):
    """A model response failed schema validation after the single repair attempt."""


class GuardrailViolation(SignalDeskError):  # noqa: N818 - the name is the contract
    """Generated content breached a guardrail and must not be surfaced."""


class AnnotationError(SignalDeskError):
    """A gold-set draw, manifest or annotation store is malformed.

    Raised rather than defaulted throughout the annotation path. A gold set is
    the one artifact in this project that cannot be regenerated, so a store that
    cannot be read has to stop the run rather than start a fresh one.
    """


class EndOfInputError(AnnotationError):
    """A key reader was asked for a keystroke and its input had ended.

    Distinct from the other annotation errors because it is the one that must
    never be answered with a default. Substituting a keystroke for absent input
    is what silently ended two annotation sessions.
    """
