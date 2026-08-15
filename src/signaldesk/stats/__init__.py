"""Disproportionality estimators.

Pure functions over NumPy arrays of ``(a, b, c, d)``. Nothing here reads the
database, the filesystem, or the settings, and nothing here logs a result. That
is what makes the four estimators independently testable against published
reference implementations, which is the only reason to believe them.

Formulas, priors and thresholds are recorded in ``docs/methodology.md`` and
locked in ``ARCHITECTURE.md`` section 8.
"""

from __future__ import annotations

from signaldesk.stats.bcpnn import information_component
from signaldesk.stats.correction import haldane_anscombe
from signaldesk.stats.mgps import fit_hyperparameters, gamma_poisson_shrinker
from signaldesk.stats.panel import MIN_CELL_A, estimate_all
from signaldesk.stats.prr import proportional_reporting_ratio, yates_chi_squared
from signaldesk.stats.ror import reporting_odds_ratio
from signaldesk.stats.types import (
    BcpnnResult,
    Contingency,
    EstimatorPanel,
    MgpsHyperparameters,
    MgpsResult,
    PrrResult,
    RorResult,
)

__all__ = [
    "MIN_CELL_A",
    "BcpnnResult",
    "Contingency",
    "EstimatorPanel",
    "MgpsHyperparameters",
    "MgpsResult",
    "PrrResult",
    "RorResult",
    "estimate_all",
    "fit_hyperparameters",
    "gamma_poisson_shrinker",
    "haldane_anscombe",
    "information_component",
    "proportional_reporting_ratio",
    "reporting_odds_ratio",
    "yates_chi_squared",
]
