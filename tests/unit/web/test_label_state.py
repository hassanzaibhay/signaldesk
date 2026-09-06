"""The parts of the label-state resolution that do not need a database.

The three-state mapping itself is exercised against real rows in
``tests/integration/test_signals_explorer.py``. What is asserted here is the
reconstruction that decides between the second and third states, because it is
the piece that can go wrong silently: a digest that drifts from the one the
ingest wrote would report every string as never-queried and the page would still
render perfectly.
"""

from __future__ import annotations

import pytest

from signaldesk.ingest.spl.scope import ScopeUnit, clean_query
from signaldesk.web.documents.models import SectionCode
from signaldesk.web.signals.labels import (
    STATE_EXPLANATIONS,
    STATE_LABELS,
    LabelState,
    _manifest_unit,
    _section_names,
    statuses_for,
)

pytestmark = pytest.mark.unit


def test_the_reconstructed_unit_matches_what_the_ingest_would_have_written() -> None:
    """The page and the ingest must agree on the key, or the state is wrong."""
    folded = "ABILIFY 10MG TABLET"
    expected = ScopeUnit(
        folded_string=folded,
        query=clean_query(folded),
        route="cleaned_string",
        ingredient_rxcui=None,
        flagged_pairs=0,
    ).manifest_unit

    assert _manifest_unit(folded) == expected
    assert _manifest_unit(folded).startswith("str:")


def test_strings_that_clean_to_one_query_share_one_unit() -> None:
    """The manifest is keyed on the query, so two spellings can be one fetch."""
    assert _manifest_unit("ABILIFY 10MG") == _manifest_unit("ABILIFY")
    assert _manifest_unit("ABILIFY.") == _manifest_unit("ABILIFY")


def test_different_drugs_do_not_collide() -> None:
    assert _manifest_unit("ABILIFY") != _manifest_unit("ABATACEPT")


def test_sections_are_named_in_the_order_a_reader_expects_them() -> None:
    """Declaration order, not alphabetical: a boxed warning leads."""
    names = _section_names({"adverse_reactions", "boxed_warning", "warnings"})

    assert names == ("Boxed warning", "Warnings", "Adverse reactions")


def test_an_unknown_section_code_is_shown_rather_than_dropped() -> None:
    """A code the model does not declare is still evidence that something is held."""
    assert _section_names({"boxed_warning", "future_section"}) == (
        "Boxed warning",
        "future_section",
    )


def test_no_sections_is_an_empty_tuple() -> None:
    assert _section_names(set()) == ()


def test_every_state_has_a_short_form_and_a_long_form() -> None:
    """A state with no wording would render as a blank cell, which is the bug."""
    for state in LabelState:
        assert STATE_LABELS[state].strip()
        assert STATE_EXPLANATIONS[state].strip()


def test_the_out_of_scope_wording_explains_the_scope_rather_than_the_drug() -> None:
    """Absence here is a limit of the fetch, and the page has to say which."""
    text = STATE_EXPLANATIONS[LabelState.OUT_OF_SCOPE]

    assert "never queried" in text
    assert "200" in text and "30,549" in text


def test_the_no_label_wording_does_not_claim_anything_about_the_drug() -> None:
    text = STATE_EXPLANATIONS[LabelState.QUERIED_NO_LABEL]

    assert "not about the drug" in text


def test_section_names_come_from_the_model_choices() -> None:
    """The page cannot drift from the vocabulary the ingest writes."""
    declared = {name for _, name in SectionCode.choices}

    assert set(_section_names({code for code, _ in SectionCode.choices})) == declared


def test_no_strings_means_no_queries() -> None:
    """Called with an empty page, this must not reach the database at all."""
    assert statuses_for([]) == {}
    assert statuses_for(["", ""]) == {}
