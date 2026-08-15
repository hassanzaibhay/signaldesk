"""Reporting odds ratio.

::

    ROR   = ad / bc
    Var(ln ROR) = 1/a + 1/b + 1/c + 1/d
    95% CI = exp( ln ROR +/- 1.96 * sqrt(Var) )

The lower bound is the reported statistic: ROR025 in the signal table. The
locked threshold is ``a >= 3 and ROR025 > 1`` (ARCHITECTURE section 8.2).

Checked against ``epitools::oddsratio.wald`` (CRAN 0.5.10.1); see
``tests/unit/stats/test_ror.py`` and ``tests/fixtures/estimators/``.
"""

from __future__ import annotations

import numpy as np

from signaldesk.stats.correction import corrected_rows, haldane_anscombe
from signaldesk.stats.types import BoolArray, Contingency, FloatArray, RorResult

#: Two-sided normal quantile for a 95 percent interval. Not 2: the interval is
#: a confidence interval and uses the exact quantile, unlike the BCPNN credible
#: bound, whose locked definition really is two standard deviations.
Z_95 = 1.959963984540054


def reporting_odds_ratio(table: Contingency, *, alpha: float = 0.05) -> RorResult:
    """ROR with a log-normal confidence interval, per row.

    ``alpha`` is exposed because the sensitivity analyses vary it; the signal
    table itself is always built at the locked 0.05.
    """
    if not 0.0 < alpha < 1.0:
        message = f"alpha must lie in (0, 1), got {alpha}"
        raise ValueError(message)

    a, b, c, d = haldane_anscombe(table)
    z = Z_95 if alpha == 0.05 else float(_normal_quantile(1.0 - alpha / 2.0))

    log_ror = np.log(a) + np.log(d) - np.log(b) - np.log(c)
    standard_error = np.sqrt(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d)

    return RorResult(
        ror=np.exp(log_ror),
        ror_lower=np.exp(log_ror - z * standard_error),
        ror_upper=np.exp(log_ror + z * standard_error),
        corrected=corrected_rows(table),
    )


def flag(table: Contingency, result: RorResult, *, min_a: int = 3) -> BoolArray:
    """The locked ROR signal rule: ``a >= min_a`` and the lower bound exceeds 1."""
    return (table.a >= min_a) & (result.ror_lower > 1.0)


def _normal_quantile(probability: float) -> FloatArray:
    """Standard normal inverse CDF.

    Imported lazily so that the common path - the locked 0.05 - does not pay for
    a SciPy import at module load.
    """
    from scipy.special import ndtri  # type: ignore[import-untyped]

    return np.asarray(ndtri(probability), dtype=np.float64)
