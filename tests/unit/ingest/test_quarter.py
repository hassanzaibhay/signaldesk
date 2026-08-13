"""The quarter value object: parsing, ordering, and stepping."""

from __future__ import annotations

import pytest

from signaldesk.ingest.faers.quarter import Quarter

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(("text", "expected"), [("2014Q3", (2014, 3)), ("2012q4", (2012, 4))])
def test_parses_either_case(text: str, expected: tuple[int, int]) -> None:
    quarter = Quarter.parse(text)
    assert (quarter.year, quarter.quarter) == expected


@pytest.mark.parametrize("text", ["2014Q5", "2014Q0", "14Q3", "2014-Q3", "", "nonsense"])
def test_rejects_malformed_input(text: str) -> None:
    with pytest.raises(ValueError, match=r"not a quarter|quarter must be"):
        Quarter.parse(text)


def test_renders_canonically() -> None:
    assert str(Quarter.parse("2012q4")) == "2012Q4"


def test_orders_across_a_year_boundary() -> None:
    assert Quarter(2013, 1) > Quarter(2012, 4)
    assert sorted([Quarter(2014, 1), Quarter(2012, 4), Quarter(2013, 2)])[0] == Quarter(2012, 4)


def test_steps_across_a_year_boundary() -> None:
    assert Quarter(2012, 4).next() == Quarter(2013, 1)
    assert Quarter(2013, 1).previous() == Quarter(2012, 4)


def test_range_is_inclusive() -> None:
    span = Quarter.range(Quarter(2012, 4), Quarter(2013, 2))
    assert [str(quarter) for quarter in span] == ["2012Q4", "2013Q1", "2013Q2"]


def test_reversed_range_is_empty_rather_than_an_error() -> None:
    assert Quarter.range(Quarter(2014, 1), Quarter(2013, 1)) == []
