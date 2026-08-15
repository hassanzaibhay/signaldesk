"""Proportional reporting ratio, with the Yates-corrected chi-squared.

::

    PRR   = [a/(a+b)] / [c/(c+d)]
    Var(ln PRR) = 1/a - 1/(a+b) + 1/c - 1/(c+d)
    95% CI = exp( ln PRR +/- 1.96 * sqrt(Var) )

    chi2  = sum over the four cells of (|O - E| - 0.5)^2 / E

The locked threshold is ``chi2 >= 4 and PRR025 > 1`` (ARCHITECTURE section 8.2).

**Two corrections, deliberately not stacked.** The ratio and its interval use
Haldane-Anscombe on rows with an empty cell, because the ratio is otherwise zero
or infinite and its variance undefined. The chi-squared does not: Yates'
continuity correction is itself the small-cell adjustment for that statistic,
and adding half a case to each cell first would shrink the same deviation twice
and depress chi2 exactly where the threshold decides. So ``chi2`` is computed
from the observed counts, always, and ``corrected`` on the result refers only to
the ratio and its bounds.

Checked against ``epitools::riskratio.wald`` (CRAN 0.5.10.1) for the ratio and
interval, and ``stats::chisq.test(correct = TRUE)`` (R 4.4.1) for the
chi-squared; see ``tests/unit/stats/test_prr.py``.
"""

from __future__ import annotations

import numpy as np

from signaldesk.stats.correction import corrected_rows, haldane_anscombe
from signaldesk.stats.ror import Z_95
from signaldesk.stats.types import BoolArray, Contingency, FloatArray, PrrResult

#: Yates' continuity correction, subtracted from each absolute deviation.
YATES = 0.5


def proportional_reporting_ratio(table: Contingency) -> PrrResult:
    """PRR, its 95 percent interval, and the Yates chi-squared, per row."""
    a, b, c, d = haldane_anscombe(table)

    log_prr = np.log(a) - np.log(a + b) - np.log(c) + np.log(c + d)
    variance = 1.0 / a - 1.0 / (a + b) + 1.0 / c - 1.0 / (c + d)
    standard_error = np.sqrt(np.clip(variance, 0.0, None))

    return PrrResult(
        prr=np.exp(log_prr),
        prr_lower=np.exp(log_prr - Z_95 * standard_error),
        prr_upper=np.exp(log_prr + Z_95 * standard_error),
        chi2=yates_chi_squared(table),
        corrected=corrected_rows(table),
    )


def yates_chi_squared(table: Contingency) -> FloatArray:
    """Pearson chi-squared with Yates' continuity correction, on observed counts.

    The four expected counts share one absolute deviation, so it is computed
    once. A row whose margins make an expected count zero has no variation to
    test and yields 0.0 rather than a division warning; such a row cannot meet
    the ``chi2 >= 4`` threshold either way.
    """
    a = table.a.astype(np.float64)
    b = table.b.astype(np.float64)
    c = table.c.astype(np.float64)
    d = table.d.astype(np.float64)
    n = a + b + c + d

    row1, row2 = a + b, c + d
    col1, col2 = a + c, b + d

    with np.errstate(divide="ignore", invalid="ignore"):
        expected = np.stack(
            [row1 * col1 / n, row1 * col2 / n, row2 * col1 / n, row2 * col2 / n], axis=0
        )
        # |ad - bc| / N is the common absolute deviation of every cell from its
        # expectation in a 2x2 table, so one Yates adjustment covers all four.
        deviation = np.clip(np.abs(a * d - b * c) / n - YATES, 0.0, None)
        statistic = (deviation**2 * np.sum(1.0 / expected, axis=0)).astype(np.float64)

    degenerate = ~np.isfinite(statistic) | np.any(expected <= 0.0, axis=0)
    return np.where(degenerate, 0.0, statistic)


def flag(result: PrrResult, *, min_chi2: float = 4.0) -> BoolArray:
    """The locked PRR signal rule: ``chi2 >= 4`` and the lower bound exceeds 1."""
    return (result.chi2 >= min_chi2) & (result.prr_lower > 1.0)
