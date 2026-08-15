"""ROR and PRR against epitools and R's chi-squared test."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from signaldesk.stats.prr import proportional_reporting_ratio, yates_chi_squared
from signaldesk.stats.ror import reporting_odds_ratio
from signaldesk.stats.types import Contingency

pytestmark = pytest.mark.unit


def _table(row: dict[str, Any]) -> Contingency:
    return Contingency(
        a=np.array([row["a"]], dtype=np.int64),
        b=np.array([row["b"]], dtype=np.int64),
        c=np.array([row["c"]], dtype=np.int64),
        d=np.array([row["d"]], dtype=np.int64),
    )


def test_ror_matches_epitools_oddsratio_wald(reference_tables: list[dict[str, Any]]) -> None:
    """Point estimate and both limits, to eight significant figures.

    Every reference table has non-zero cells, so no continuity correction is in
    play here and the comparison is exact arithmetic against exact arithmetic.
    """
    for row in reference_tables:
        result = reporting_odds_ratio(_table(row))
        assert not bool(result.corrected[0]), row["name"]
        for field, expected in (
            (result.ror, row["ror"]),
            (result.ror_lower, row["ror_lower"]),
            (result.ror_upper, row["ror_upper"]),
        ):
            assert float(field[0]) == pytest.approx(expected, rel=1e-8), row["name"]


def test_prr_matches_epitools_riskratio_wald(reference_tables: list[dict[str, Any]]) -> None:
    for row in reference_tables:
        result = proportional_reporting_ratio(_table(row))
        for field, expected in (
            (result.prr, row["prr"]),
            (result.prr_lower, row["prr_lower"]),
            (result.prr_upper, row["prr_upper"]),
        ):
            assert float(field[0]) == pytest.approx(expected, rel=1e-8), row["name"]


def test_chi_squared_matches_r_with_yates(reference_tables: list[dict[str, Any]]) -> None:
    for row in reference_tables:
        statistic = yates_chi_squared(_table(row))
        assert float(statistic[0]) == pytest.approx(row["chi2_yates"], rel=1e-8), row["name"]


def test_chi_squared_ignores_the_haldane_anscombe_correction() -> None:
    """A zero cell corrects the ratio but must leave the chi-squared alone.

    Yates already adjusts this statistic for small cells. If the correction
    reached it too, the same deviation would be shrunk twice and the ``chi2 >= 4``
    threshold would be applied to a number that is not the Yates statistic.
    """
    zero_cell = Contingency(
        a=np.array([0], dtype=np.int64),
        b=np.array([500], dtype=np.int64),
        c=np.array([300], dtype=np.int64),
        d=np.array([99200], dtype=np.int64),
    )
    result = proportional_reporting_ratio(zero_cell)

    assert bool(result.corrected[0])
    assert np.isfinite(result.prr[0])
    # Computed from the observed counts: a=0 gives |ad - bc|/N = 1.5 exactly.
    assert float(result.chi2[0]) == pytest.approx(float(yates_chi_squared(zero_cell)[0]))
    assert float(result.chi2[0]) > 0.0


def test_zero_cell_is_corrected_and_flagged_rather_than_infinite() -> None:
    table = Contingency(
        a=np.array([5, 0], dtype=np.int64),
        b=np.array([0, 500], dtype=np.int64),
        c=np.array([20, 300], dtype=np.int64),
        d=np.array([9975, 99200], dtype=np.int64),
    )
    result = reporting_odds_ratio(table)

    assert result.corrected.tolist() == [True, True]
    assert np.all(np.isfinite(result.ror))
    assert np.all(np.isfinite(result.ror_lower))
    assert np.all(np.isfinite(result.ror_upper))


def test_rows_without_a_zero_cell_are_not_corrected() -> None:
    """The correction must not touch a table that does not need it.

    Applied unconditionally, +0.5 biases every estimate toward the null. This is
    the test that would fail if someone moved the adjustment outside the mask.
    """
    table = Contingency(
        a=np.array([25], dtype=np.int64),
        b=np.array([1975], dtype=np.int64),
        c=np.array([1000], dtype=np.int64),
        d=np.array([97000], dtype=np.int64),
    )
    result = reporting_odds_ratio(table)

    assert not bool(result.corrected[0])
    assert float(result.ror[0]) == pytest.approx((25 * 97000) / (1975 * 1000), rel=1e-12)
