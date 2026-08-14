"""Normalization: units, partial dates, sentinels, and derived seriousness.

These are the functions that decide what a number means, so each conversion is
checked against a hand-computed value rather than against itself.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from signaldesk.core.errors import SchemaMismatchError
from signaldesk.ingest.faers.parse import (
    DATE_PRECISION_DAY,
    DATE_PRECISION_MISSING,
    DATE_PRECISION_MONTH,
    DATE_PRECISION_YEAR,
    age_in_years,
    assert_known_age_units,
    date_precision,
    derive_seriousness,
    normalize_country,
    normalize_term,
    parse_partial_date,
    weight_in_kg,
)
from signaldesk.ingest.faers.quarter import Quarter

pytestmark = pytest.mark.unit


def _ages(pairs: list[tuple[str | None, str | None]]) -> list[float | None]:
    frame = pl.DataFrame(
        {"age": [value for value, _ in pairs], "age_cod": [unit for _, unit in pairs]},
        schema={"age": pl.String, "age_cod": pl.String},
    )
    return frame.select(age_in_years(pl.col("age"), pl.col("age_cod")).alias("y"))["y"].to_list()


@pytest.mark.parametrize(
    ("value", "unit", "expected"),
    [
        ("5", "DEC", 50.0),
        ("42", "YR", 42.0),
        ("18", "MON", 1.5),
        ("52", "WK", 52 / 52.1775),
        ("365", "DY", 365 / 365.25),
        ("24", "HR", 24 / 8766.0),
        ("60", "MIN", 60 / 525960.0),
        ("30", "SEC", 30 / 31_557_600.0),
    ],
)
def test_every_age_unit_converts_to_years(value: str, unit: str, expected: float) -> None:
    assert _ages([(value, unit)])[0] == pytest.approx(expected)


@pytest.mark.parametrize(
    ("value", "unit"),
    [
        ("999", "YR"),  # older than any human
        ("13", "DEC"),  # 130 years, over the ceiling
        ("-5", "YR"),  # negative
        ("42", None),  # a number with no unit is not assumed to be years
        (None, "YR"),  # no value
        ("nonsense", "YR"),  # free text in a numeric column
    ],
)
def test_implausible_or_unusable_ages_become_null(value: str | None, unit: str | None) -> None:
    assert _ages([(value, unit)])[0] is None


def test_age_boundary_is_inclusive_at_the_ceiling() -> None:
    assert _ages([("120", "YR")])[0] == pytest.approx(120.0)


def test_unknown_age_unit_raises_rather_than_nulling() -> None:
    frame = pl.DataFrame({"age_cod": ["YR", "FORTNIGHT"]})
    with pytest.raises(SchemaMismatchError, match="FORTNIGHT"):
        assert_known_age_units(frame, Quarter.parse("2013Q1"))


@pytest.mark.parametrize(
    ("value", "unit", "expected"),
    [("70", "KG", 70.0), ("154", "LBS", 154 * 0.45359237), ("3200", "GMS", 3.2)],
)
def test_weight_units_convert_to_kilograms(value: str, unit: str, expected: float) -> None:
    frame = pl.DataFrame({"wt": [value], "wt_cod": [unit]})
    result = frame.select(weight_in_kg(pl.col("wt"), pl.col("wt_cod")).alias("kg"))["kg"][0]
    assert result == pytest.approx(expected)


@pytest.mark.parametrize(("value", "unit"), [("70", "YEARS"), ("0", "KG"), ("900", "KG")])
def test_corrupt_or_implausible_weights_become_null(value: str, unit: str) -> None:
    frame = pl.DataFrame({"wt": [value], "wt_cod": [unit]})
    assert frame.select(weight_in_kg(pl.col("wt"), pl.col("wt_cod")).alias("kg"))["kg"][0] is None


@pytest.mark.parametrize(
    ("raw", "expected_date", "expected_precision"),
    [
        ("20130215", date(2013, 2, 15), DATE_PRECISION_DAY),
        ("201302", date(2013, 2, 1), DATE_PRECISION_MONTH),
        ("2013", date(2013, 1, 1), DATE_PRECISION_YEAR),
        (None, None, DATE_PRECISION_MISSING),
        ("", None, DATE_PRECISION_MISSING),
        ("2013021", None, DATE_PRECISION_MISSING),
    ],
)
def test_partial_dates_keep_their_precision(
    raw: str | None, expected_date: date | None, expected_precision: str
) -> None:
    frame = pl.DataFrame({"raw": [raw]}, schema={"raw": pl.String})
    result = frame.select(
        parse_partial_date(pl.col("raw")).alias("d"),
        date_precision(pl.col("raw")).alias("p"),
    )
    assert result["d"][0] == expected_date
    assert result["p"][0] == expected_precision


def test_impossible_calendar_date_is_null_but_keeps_its_stated_precision() -> None:
    frame = pl.DataFrame({"raw": ["20130231"]})
    result = frame.select(
        parse_partial_date(pl.col("raw")).alias("d"), date_precision(pl.col("raw")).alias("p")
    )
    assert result["d"][0] is None
    assert result["p"][0] == DATE_PRECISION_DAY


def test_reaction_terms_are_uppercased_and_whitespace_collapsed_only() -> None:
    frame = pl.DataFrame({"pt": ["  drug   hypersensitivity ", "nausea", ""]})
    result = frame.select(normalize_term(pl.col("pt")).alias("pt"))["pt"].to_list()
    assert result == ["DRUG HYPERSENSITIVITY", "NAUSEA", None]


def test_country_is_trimmed_and_uppercased() -> None:
    frame = pl.DataFrame({"c": [" us ", "gb"]})
    assert frame.select(normalize_country(pl.col("c")).alias("c"))["c"].to_list() == ["US", "GB"]


def test_stated_unknown_country_becomes_null() -> None:
    # Left as a value it would form one large block of records sharing only
    # absent information, and deduplication blocks on country.
    frame = pl.DataFrame({"c": ["COUNTRY NOT SPECIFIED", "US"]})
    assert frame.select(normalize_country(pl.col("c")).alias("c"))["c"].to_list() == [None, "US"]


def test_seriousness_is_derived_from_outcome_codes() -> None:
    outcomes = pl.DataFrame(
        {
            "primaryid": [1, 1, 2, 3],
            "outc_cod": ["HO", "DE", "OT", "CA"],
        }
    )
    derived = derive_seriousness(outcomes).sort("primaryid")
    assert derived["serious"].to_list() == [True, True, True]
    assert derived["serious_death"].to_list() == [True, False, False]
    assert derived["outcome_codes"][0].to_list() == ["DE", "HO"]
