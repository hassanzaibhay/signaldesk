"""The error hierarchy is part of the public contract: callers catch by category."""

from __future__ import annotations

import pytest

from signaldesk.core import errors

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("error", "parent"),
    [
        (errors.IngestError, errors.SignalDeskError),
        (errors.SchemaMismatchError, errors.IngestError),
        (errors.NormalizationError, errors.IngestError),
        (errors.EstimatorConvergenceError, errors.SignalDeskError),
        (errors.RetrievalError, errors.SignalDeskError),
        (errors.ProviderError, errors.SignalDeskError),
        (errors.RateLimitError, errors.ProviderError),
        (errors.StructuredOutputError, errors.ProviderError),
        (errors.GuardrailViolation, errors.SignalDeskError),
    ],
)
def test_error_inherits_from_its_category(error: type[Exception], parent: type[Exception]) -> None:
    assert issubclass(error, parent)


def test_catching_the_base_class_catches_a_provider_failure() -> None:
    with pytest.raises(errors.SignalDeskError):
        raise errors.RateLimitError("slow down")
