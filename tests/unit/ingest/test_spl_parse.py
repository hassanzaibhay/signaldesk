"""Sectioning openFDA label results.

The fixtures are hand-built to the documented response shape, not captured from
the service: this prompt is not permitted to call openFDA, and inventing a
recorded response would be worse than declaring a constructed one. They encode
the awkward cases the parser has to survive - a label with no `openfda` block, a
section array with blank elements, two revisions of one `set_id` in a single
page - rather than a happy path. They are marked for replacement with captured
bodies once a real run happens.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from signaldesk.ingest.spl.parse import (
    SplParseError,
    parse_result,
    parse_results,
    sections_of,
)

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "spl"


def _load(name: str) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return payload


@pytest.fixture
def full() -> dict[str, Any]:
    return _load("label_full.json")["results"][0]


@pytest.fixture
def minimal() -> dict[str, Any]:
    return _load("label_minimal.json")["results"][0]


def test_identity_and_openfda_fields_are_carried(full: dict[str, Any]) -> None:
    record = parse_result(full)
    assert record.set_id == "6a1b2c3d-0000-4a11-9f00-000000000001"
    assert record.spl_id == "aaaa1111-bbbb-2222-cccc-333333333333"
    assert record.version == "7"
    assert record.effective_time == "20240118"
    assert record.brand_names == ["EXAMPLAMAB"]
    assert record.rxcuis == ["1234567", "1234568"]


def test_every_element_of_a_section_array_becomes_its_own_row(full: dict[str, Any]) -> None:
    """The elements are separate blocks, not one paragraph split by accident."""
    sections = sections_of(full)
    boxed = [item for item in sections if item.section_code == "boxed_warning"]
    assert [item.ordinal for item in boxed] == [0, 1]
    assert "RISK OF SERIOUS INFECTION" in boxed[0].text
    assert "increased risk" in boxed[1].text


def test_sections_are_emitted_in_a_deterministic_order(full: dict[str, Any]) -> None:
    codes = [item.section_code for item in sections_of(full)]
    assert codes == [
        "boxed_warning",
        "boxed_warning",
        "warnings_and_cautions",
        "warnings_and_cautions",
        "adverse_reactions",
        "adverse_reactions",
    ]


def test_a_label_with_no_openfda_block_still_parses(minimal: dict[str, Any]) -> None:
    """Older and unapproved labels carry no openfda block. They are not errors."""
    record = parse_result(minimal)
    assert record.set_id.endswith("0002")
    assert record.brand_names == []
    assert record.rxcuis == []
    assert record.spl_id == ""


def test_blank_and_empty_section_elements_are_dropped(minimal: dict[str, Any]) -> None:
    record = parse_result(minimal)
    codes = [item.section_code for item in record.sections]
    assert codes == ["warnings"]
    assert record.sections[0].ordinal == 0
    assert record.sections[0].text.startswith("WARNINGS Anaphylactoid")


def test_the_older_warnings_field_is_kept_separate_from_warnings_and_cautions(
    full: dict[str, Any], minimal: dict[str, Any]
) -> None:
    """Both eras coexist in the corpus, so they are distinct codes, not merged."""
    assert {item.section_code for item in sections_of(minimal)} == {"warnings"}
    assert "warnings" not in {item.section_code for item in sections_of(full)}
    assert "warnings_and_cautions" in {item.section_code for item in sections_of(full)}


def test_a_result_without_a_set_id_is_an_error() -> None:
    """Without it a label cannot be stored idempotently or revisited."""
    with pytest.raises(SplParseError, match="no set_id"):
        parse_result({"adverse_reactions": ["something"]})


def test_only_the_newest_revision_of_a_set_id_survives_a_page() -> None:
    """One page can hold several revisions; storing all of them would leave the
    retrieval layer choosing a version at read time."""
    results = _load("label_revisions.json")["results"]
    records = parse_results(results)
    assert len(records) == 1
    assert records[0].effective_time == "20250612"
    assert records[0].spl_id == "new-revision"
    assert records[0].sections[0].text == "6 ADVERSE REACTIONS Nausea was reported."


def test_parse_results_is_stable_regardless_of_input_order() -> None:
    results = _load("label_revisions.json")["results"]
    assert parse_results(results) == parse_results(list(reversed(results)))
