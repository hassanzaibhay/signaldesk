"""Schema declaration and header validation.

The header check is the thing that catches an upstream change before it becomes
a wrong number, so its failure message is tested as carefully as its success.
"""

from __future__ import annotations

import pytest

from signaldesk.core.errors import SchemaMismatchError
from signaldesk.ingest.faers.quarter import Quarter
from signaldesk.ingest.faers.schemas import (
    AGE_UNIT_YEARS,
    Era,
    Table,
    era_for,
    schema_for,
    validate_header,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("2012Q4", Era.FIRST),
        ("2013Q1", Era.ORIGINAL),
        ("2014Q2", Era.ORIGINAL),
        ("2014Q3", Era.CURRENT),
        ("2026Q2", Era.CURRENT),
    ],
)
def test_era_boundaries_match_the_observed_headers(label: str, expected: Era) -> None:
    assert era_for(Quarter.parse(label)) == expected


def test_sex_column_is_renamed_across_the_boundary() -> None:
    before = schema_for(Table.DEMO, Quarter.parse("2013Q1"))
    after = schema_for(Table.DEMO, Quarter.parse("2014Q3"))
    assert "gndr_cod" in before.source_names
    assert "sex" in after.source_names
    # The whole point of the mapping: one canonical name either side.
    assert "sex" in before.canonical_names
    assert "sex" in after.canonical_names


def test_first_quarter_spells_two_columns_differently() -> None:
    drug = schema_for(Table.DRUG, Quarter.parse("2012Q4"))
    outcome = schema_for(Table.OUTC, Quarter.parse("2012Q4"))
    assert "lot_nbr" in drug.source_names
    assert "outc_code" in outcome.source_names
    # Canonical names are shared with every other quarter.
    assert "lot_num" in drug.canonical_names
    assert "outc_cod" in outcome.canonical_names


def test_prod_ai_exists_only_in_the_current_era() -> None:
    assert "prod_ai" not in schema_for(Table.DRUG, Quarter.parse("2013Q1")).canonical_names
    assert "prod_ai" in schema_for(Table.DRUG, Quarter.parse("2014Q3")).canonical_names


def test_matching_header_passes() -> None:
    quarter = Quarter.parse("2013Q1")
    schema = schema_for(Table.REAC, quarter)
    validate_header(schema, list(schema.source_names), quarter)


def test_mismatch_names_both_directions() -> None:
    quarter = Quarter.parse("2013Q1")
    schema = schema_for(Table.OUTC, quarter)
    with pytest.raises(SchemaMismatchError) as failure:
        validate_header(schema, ["primaryid", "caseid", "outc_code"], quarter)

    message = str(failure.value)
    assert "2013Q1" in message
    assert "OUTC" in message
    assert "outc_cod" in message
    assert "outc_code" in message


def test_mismatch_reports_a_reordering() -> None:
    quarter = Quarter.parse("2013Q1")
    schema = schema_for(Table.OUTC, quarter)
    with pytest.raises(SchemaMismatchError, match="different order"):
        validate_header(schema, ["caseid", "primaryid", "outc_cod"], quarter)


def test_header_is_compared_after_trimming() -> None:
    quarter = Quarter.parse("2013Q1")
    schema = schema_for(Table.OUTC, quarter)
    validate_header(schema, ["primaryid ", " caseid", "outc_cod "], quarter)


def test_every_observed_age_unit_is_declared() -> None:
    # SEC was found by the parser refusing to guess, on 2012Q4.
    assert set(AGE_UNIT_YEARS) == {"DEC", "YR", "MON", "WK", "DY", "HR", "MIN", "SEC"}
