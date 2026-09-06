"""How a cell renders a number, and what it says when there is not one.

A blank cell is indistinguishable from a rendering bug, so every path through
these filters produces words. The precision rules matter for the same reason:
the largest ROR in the corpus is above 9.6e7 and the smallest useful one is near
1, and one format string cannot serve both.
"""

from __future__ import annotations

import math

import pytest

from signaldesk.web.signals.templatetags.signal_format import (
    NOT_ESTIMABLE,
    count,
    estimator,
    interval,
)

pytestmark = pytest.mark.unit


def test_counts_carry_thousands_separators() -> None:
    assert count(2785896) == "2,785,896"
    assert count(0) == "0"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1.2678, "1.27"),
        (12.5, "12.50"),
        (99.994, "99.99"),
        (616.0230614638616, "616.0"),
        (99999.4, "99,999.4"),
        (96329924.99999999, "9.63e+07"),
    ],
)
def test_estimators_stay_readable_across_seven_orders_of_magnitude(
    value: float, expected: str
) -> None:
    assert estimator(value) == expected


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -math.inf, "not a number"])
def test_anything_that_is_not_a_finite_number_says_so_in_words(value: object) -> None:
    """Never a blank, never a dash, never a zero standing in for a missing value."""
    assert estimator(value) == NOT_ESTIMABLE


def test_a_count_that_is_not_a_number_says_so_too() -> None:
    assert count(None) == NOT_ESTIMABLE


def test_an_interval_needs_both_ends() -> None:
    assert interval(11.0, 14.0) == "11.00 to 14.00"


@pytest.mark.parametrize(("low", "high"), [(None, 14.0), (11.0, None), (None, None)])
def test_half_an_interval_is_not_an_interval(low: object, high: object) -> None:
    """One bound alone would be read as the estimate. Neither end is shown."""
    assert interval(low, high) == "no interval"
