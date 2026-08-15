"""Properties the estimators must hold for any table, and the degenerate cases.

The golden tests pin the arithmetic against published implementations on seven
tables. These cover the space between and either side of them.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from signaldesk.stats.bcpnn import information_component
from signaldesk.stats.correction import haldane_anscombe
from signaldesk.stats.prr import proportional_reporting_ratio, yates_chi_squared
from signaldesk.stats.ror import reporting_odds_ratio
from signaldesk.stats.types import Contingency

pytestmark = pytest.mark.unit

# Wide enough to reach both the sparse corner where the corrections matter and
# counts large enough to lose precision if the arithmetic were done on ratios
# rather than in logs.
cells = st.integers(min_value=0, max_value=10**9)

PROPERTY_SETTINGS = settings(
    max_examples=300,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


def _one(a: int, b: int, c: int, d: int) -> Contingency:
    return Contingency(
        a=np.array([a], dtype=np.int64),
        b=np.array([b], dtype=np.int64),
        c=np.array([c], dtype=np.int64),
        d=np.array([d], dtype=np.int64),
    )


@PROPERTY_SETTINGS
@given(a=cells, b=cells, c=cells, d=cells)
def test_ror_interval_brackets_its_point_estimate(a: int, b: int, c: int, d: int) -> None:
    assume(a + b + c + d > 0)
    result = reporting_odds_ratio(_one(a, b, c, d))
    assert float(result.ror_lower[0]) <= float(result.ror[0]) <= float(result.ror_upper[0])


@PROPERTY_SETTINGS
@given(a=cells, b=cells, c=cells, d=cells)
def test_prr_interval_brackets_its_point_estimate(a: int, b: int, c: int, d: int) -> None:
    assume(a + b + c + d > 0)
    result = proportional_reporting_ratio(_one(a, b, c, d))
    assert float(result.prr_lower[0]) <= float(result.prr[0]) <= float(result.prr_upper[0])


@PROPERTY_SETTINGS
@given(a=cells, b=cells, c=cells, d=cells)
def test_ic025_never_exceeds_ic(a: int, b: int, c: int, d: int) -> None:
    assume(a + b + c + d > 0)
    result = information_component(_one(a, b, c, d))
    assert float(result.ic025[0]) <= float(result.ic[0])
    assert float(result.ic025_exact[0]) <= float(result.ic_posterior_mean[0])


@PROPERTY_SETTINGS
@given(a=cells, b=cells, c=cells, d=cells)
def test_no_estimator_returns_nan(a: int, b: int, c: int, d: int) -> None:
    """A NaN in a statistics pipeline is a fabricated result that looks like data.

    Infinities are permitted where a statistic genuinely diverges - the locked
    IC at ``a = 0`` is one - but a NaN says the arithmetic went somewhere its
    author did not intend.
    """
    assume(a + b + c + d > 0)
    table = _one(a, b, c, d)

    ror = reporting_odds_ratio(table)
    prr = proportional_reporting_ratio(table)
    bcpnn = information_component(table)

    for array in (
        ror.ror,
        ror.ror_lower,
        ror.ror_upper,
        prr.prr,
        prr.prr_lower,
        prr.prr_upper,
        prr.chi2,
        bcpnn.ic,
        bcpnn.ic025,
        bcpnn.ic_variance,
        bcpnn.ic_posterior_mean,
        bcpnn.ic025_exact,
    ):
        assert not np.any(np.isnan(array)), (a, b, c, d)


@PROPERTY_SETTINGS
@given(
    a=st.integers(min_value=1, max_value=10**6),
    b=st.integers(min_value=1, max_value=10**6),
    c=st.integers(min_value=1, max_value=10**6),
    d=st.integers(min_value=1, max_value=10**6),
)
def test_ror_is_invariant_under_transposing_the_table(a: int, b: int, c: int, d: int) -> None:
    """The odds ratio does not care which margin is the exposure.

    Swapping ``b`` and ``c`` transposes the table. If the implementation ever
    read the cells in a different order than the CI does, this is what catches
    it.
    """
    direct = reporting_odds_ratio(_one(a, b, c, d))
    transposed = reporting_odds_ratio(_one(a, c, b, d))

    assert float(direct.ror[0]) == pytest.approx(float(transposed.ror[0]), rel=1e-12)
    assert float(direct.ror_lower[0]) == pytest.approx(float(transposed.ror_lower[0]), rel=1e-12)


@PROPERTY_SETTINGS
@given(
    a=st.integers(min_value=1, max_value=10**4),
    b=st.integers(min_value=1, max_value=10**4),
    c=st.integers(min_value=1, max_value=10**4),
    d=st.integers(min_value=1, max_value=10**4),
    scale=st.integers(min_value=2, max_value=50),
)
def test_scaling_every_cell_narrows_the_interval_without_moving_the_point(
    a: int, b: int, c: int, d: int, scale: int
) -> None:
    """More of the same evidence: same ratio, tighter bounds."""
    small = reporting_odds_ratio(_one(a, b, c, d))
    large = reporting_odds_ratio(_one(a * scale, b * scale, c * scale, d * scale))

    assert float(large.ror[0]) == pytest.approx(float(small.ror[0]), rel=1e-9)
    small_width = float(small.ror_upper[0]) - float(small.ror_lower[0])
    large_width = float(large.ror_upper[0]) - float(large.ror_lower[0])
    assert large_width < small_width


@PROPERTY_SETTINGS
@given(
    a=st.integers(min_value=1, max_value=10**5),
    b=st.integers(min_value=1, max_value=10**5),
    c=st.integers(min_value=1, max_value=10**5),
    d=st.integers(min_value=1, max_value=10**5),
    extra=st.integers(min_value=1, max_value=10**4),
)
def test_ratio_estimators_increase_with_the_co_reported_count(
    a: int, b: int, c: int, d: int, extra: int
) -> None:
    """Holding ``b``, ``c`` and ``d`` fixed, more co-reports means a larger ratio.

    Unconditional for ROR and PRR: their denominators do not contain ``a``.
    """
    lower = _one(a, b, c, d)
    higher = _one(a + extra, b, c, d)

    assert float(reporting_odds_ratio(higher).ror[0]) > float(reporting_odds_ratio(lower).ror[0])
    assert float(proportional_reporting_ratio(higher).prr[0]) > float(
        proportional_reporting_ratio(lower).prr[0]
    )


@PROPERTY_SETTINGS
@given(
    a=st.integers(min_value=1, max_value=100),
    b=st.integers(min_value=10**6, max_value=10**7),
    c=st.integers(min_value=10**6, max_value=10**7),
    d=st.integers(min_value=10**8, max_value=10**9),
    extra=st.integers(min_value=1, max_value=100),
)
def test_ic_increases_with_the_co_reported_count_when_a_is_small_beside_its_margins(
    a: int, b: int, c: int, d: int, extra: int
) -> None:
    """IC is monotonic in ``a`` only while ``a`` stays small beside ``b`` and ``c``.

    ``d(log IC)/da = 1/N + 1/a - 1/(a+b) - 1/(a+c)``, which is positive exactly
    when ``1/a`` still dominates the two marginal terms. That holds throughout
    the regime this corpus lives in - a drug with a million reports, a term with
    a million reports, and a hundred cases naming both - and fails outside it.
    See the counterexample below.
    """
    lower = _one(a, b, c, d)
    higher = _one(a + extra, b, c, d)

    assert float(information_component(higher).ic[0]) > float(information_component(lower).ic[0])


@pytest.mark.parametrize(
    ("cells_low", "cells_high"),
    [
        ((2, 1, 1, 2), (3, 1, 1, 2)),
        ((1, 1, 1, 10_000_000), (2, 1, 1, 10_000_000)),
    ],
)
def test_ic_is_not_monotonic_in_a_when_a_rivals_its_margins(
    cells_low: tuple[int, int, int, int], cells_high: tuple[int, int, int, int]
) -> None:
    """The counterexamples, pinned so the condition above cannot be dropped.

    ``a`` appears in ``N`` and in both marginals, so a further co-report also
    enlarges the expectation it is measured against. Once ``a`` is comparable to
    ``b`` and ``c``, ``(a+b)(a+c)`` grows about as ``a^2`` while ``N*a`` grows as
    ``a``, and IC falls.

    A large background does not rescue it: the second case has ``d`` at ten
    million and still decreases. It is the marginals that matter, not the
    corpus size, which is why the property above is conditioned on ``b`` and
    ``c`` rather than on ``d``.
    """
    low = float(information_component(_one(*cells_low)).ic[0])
    high = float(information_component(_one(*cells_high)).ic[0])

    assert high < low


@PROPERTY_SETTINGS
@given(a=cells, b=cells, c=cells, d=cells)
def test_correction_applies_exactly_to_the_rows_with_an_empty_cell(
    a: int, b: int, c: int, d: int
) -> None:
    table = _one(a, b, c, d)
    corrected_a, corrected_b, corrected_c, corrected_d = haldane_anscombe(table)
    expected_shift = 0.5 if min(a, b, c, d) == 0 else 0.0

    assert float(corrected_a[0]) == a + expected_shift
    assert float(corrected_b[0]) == b + expected_shift
    assert float(corrected_c[0]) == c + expected_shift
    assert float(corrected_d[0]) == d + expected_shift


def test_all_zero_table_is_rejected_rather_than_scored() -> None:
    """An empty background has no statistic, and pretending otherwise is worse.

    ``N = 0`` makes every marginal probability 0/0. The table type does not
    forbid it, so the behaviour is pinned here: the ratios come back as
    infinities or the correction's neutral value, never as a finite number that
    would read as a measurement.
    """
    table = _one(0, 0, 0, 0)
    result = reporting_odds_ratio(table)

    assert bool(result.corrected[0])
    assert float(result.ror[0]) == pytest.approx(1.0)


def test_counts_near_the_int64_limit_do_not_overflow() -> None:
    """``a*d`` and ``b*c`` would overflow int64; the logs must not.

    This is why ROR is computed as a difference of logs rather than as the
    ratio ``ad/bc``. On the naive form these cells wrap around to a negative
    product and the result is a plausible-looking number of the wrong sign.
    """
    big = 3 * 10**9
    table = _one(big, big // 2, big // 3, big * 2)
    result = reporting_odds_ratio(table)

    assert np.isfinite(result.ror[0])
    assert float(result.ror[0]) == pytest.approx(
        (big * (big * 2)) / ((big // 2) * (big // 3)), rel=1e-9
    )


def test_single_report_pair_is_computed_not_dropped() -> None:
    """One case is a legitimate table. It must score, and score weakly."""
    table = _one(1, 999, 1000, 998000)
    ror = reporting_odds_ratio(table)
    bcpnn = information_component(table)

    assert np.isfinite(ror.ror[0])
    assert float(ror.ror_lower[0]) < float(ror.ror[0])
    assert float(bcpnn.ic025[0]) < float(bcpnn.ic[0])


def test_chi_squared_is_zero_when_a_margin_is_empty() -> None:
    """No variation to test, so no statistic, and certainly not a signal."""
    assert float(yates_chi_squared(_one(0, 0, 500, 99500))[0]) == 0.0
    assert float(yates_chi_squared(_one(0, 500, 0, 99500))[0]) == 0.0


def test_mismatched_cell_shapes_are_rejected() -> None:
    with pytest.raises(ValueError, match="share a shape"):
        Contingency(
            a=np.array([1, 2], dtype=np.int64),
            b=np.array([1], dtype=np.int64),
            c=np.array([1], dtype=np.int64),
            d=np.array([1], dtype=np.int64),
        )


def test_negative_counts_are_rejected() -> None:
    with pytest.raises(ValueError, match="negative count in cell c"):
        Contingency(
            a=np.array([1], dtype=np.int64),
            b=np.array([1], dtype=np.int64),
            c=np.array([-1], dtype=np.int64),
            d=np.array([1], dtype=np.int64),
        )
