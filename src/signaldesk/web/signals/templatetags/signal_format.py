"""Number formatting for the signals table.

In Python rather than in the template, for two reasons. ``humanize`` is not
installed and adding it to ``INSTALLED_APPS`` to get thousands separators would
be a settings change for a comma. And the rules here are not cosmetic: a ratio
of 96,329,925 and a ratio of 1.27 are both real values in this table, and a
single format string renders one of them unreadable. What a cell says when there
is no value is a correctness question, not a styling one, so it is decided here
once.
"""

from __future__ import annotations

from django import template

register = template.Library()

#: What a cell says when the estimator produced nothing usable - a null, a NaN,
#: or an infinity from a zero cell. Words, not a blank and not a dash: a blank
#: cell is indistinguishable from a rendering bug, which is exactly the failure
#: this page is trying not to have.
NOT_ESTIMABLE = "not estimable"


@register.filter
def count(value: object) -> str:
    """An integer with thousands separators."""
    try:
        return f"{int(value):,}"  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return NOT_ESTIMABLE


@register.filter
def estimator(value: object) -> str:
    """One estimator value, at a precision that stays readable across the range.

    Small-cell pairs produce enormous ratios - the largest ROR in this run is
    above 9.6e7, from a 2x2 table with a zero cell that the Haldane-Anscombe
    correction has moved off zero. Printed in full those digits crowd out every
    other column and imply a precision the estimate does not have, so anything
    at or above 100,000 is shown in scientific notation.
    """
    if value is None:
        return NOT_ESTIMABLE
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return NOT_ESTIMABLE
    if number != number or abs(number) == float("inf"):
        return NOT_ESTIMABLE
    if abs(number) >= 100_000:
        return f"{number:.2e}"
    if abs(number) >= 100:
        return f"{number:,.1f}"
    return f"{number:.2f}"


@register.simple_tag
def interval(lower: object, upper: object) -> str:
    """A confidence interval, or a statement that there is not one.

    Both ends or neither. Half an interval is not an interval, and rendering one
    bound on its own would invite it to be read as the estimate.
    """
    low = estimator(lower)
    high = estimator(upper)
    if NOT_ESTIMABLE in (low, high):
        return "no interval"
    return f"{low} to {high}"
