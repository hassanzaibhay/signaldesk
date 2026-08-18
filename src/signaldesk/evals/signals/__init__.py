"""Reference-set validation of the disproportionality estimators.

ARCHITECTURE section 8.3. Metrics are computed per method per published
standard, with every excluded pair counted and reported beside the result.
"""

from __future__ import annotations

from signaldesk.evals.signals.reference import (
    Label,
    OutcomeMap,
    ReferencePair,
    ReferenceSet,
    ReferenceSetError,
    load_all,
    load_outcome_map,
    load_reference_set,
    reference_root,
)

__all__ = [
    "Label",
    "OutcomeMap",
    "ReferencePair",
    "ReferenceSet",
    "ReferenceSetError",
    "load_all",
    "load_outcome_map",
    "load_reference_set",
    "reference_root",
]
