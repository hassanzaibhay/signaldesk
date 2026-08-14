"""Writing normalization results, and re-writing them.

Normalization is re-run whenever the cache warms, an override is added or a
cleaning rule changes, so the write has to converge on the same rows rather than
accumulate. The combination case is where an appending write would show up first:
one string, several ingredient rows.
"""

from __future__ import annotations

import pytest

from signaldesk.ingest.rxnorm import match
from signaldesk.ingest.rxnorm.client import Concept
from signaldesk.ingest.rxnorm.store import store_matches
from signaldesk.web.signals.models import DrugConcept, DrugStringIngredient, DrugStringMatch

pytestmark = [pytest.mark.integration, pytest.mark.django_db(transaction=True)]


def _match(**overrides: object) -> match.StringMatch:
    fields: dict[str, object] = {
        "source_field": "drugname",
        "folded": "ZANTAC",
        "cleaned": "ZANTAC",
        "method": match.METHOD_EXACT,
        "rxcui": "152523",
        "ingredients": (Concept(rxcui="9143", name="ranitidine", tty="IN"),),
        "components_total": 1,
        "components_matched": 1,
    }
    fields.update(overrides)
    return match.StringMatch(**fields)  # type: ignore[arg-type]


def _combination() -> match.StringMatch:
    return _match(
        folded="HYDROCODONE BITARTRATE AND ACETAMINOPHEN",
        cleaned="HYDROCODONE BITARTRATE AND ACETAMINOPHEN",
        rxcui=None,
        ingredients=(
            Concept(rxcui="5489", name="hydrocodone", tty="IN"),
            Concept(rxcui="161", name="acetaminophen", tty="IN"),
        ),
        components_total=2,
        components_matched=2,
    )


def test_a_match_and_its_ingredient_are_written() -> None:
    counts = store_matches([_match()])

    assert counts == type(counts)(strings=1, concepts=1, associations=1)
    row = DrugStringMatch.objects.get(source_field="drugname", folded_string="ZANTAC")
    assert row.rxcui == 152523
    assert row.match_method == match.METHOD_EXACT
    assert [item.ingredient_id for item in row.ingredients.all()] == [9143]


def test_a_combination_writes_one_row_per_ingredient_in_order() -> None:
    store_matches([_combination()])

    row = DrugStringMatch.objects.get(folded_string="HYDROCODONE BITARTRATE AND ACETAMINOPHEN")
    assert row.rxcui is None
    assert [item.ingredient_id for item in row.ingredients.order_by("ordinal")] == [5489, 161]
    assert row.components_total == 2


def test_writing_twice_converges_rather_than_doubling() -> None:
    """The property that makes normalization re-runnable."""
    store_matches([_match(), _combination()])
    first = (
        DrugStringMatch.objects.count(),
        DrugStringIngredient.objects.count(),
        DrugConcept.objects.count(),
    )

    store_matches([_match(), _combination()])

    assert (
        DrugStringMatch.objects.count(),
        DrugStringIngredient.objects.count(),
        DrugConcept.objects.count(),
    ) == first


def test_a_corrected_mapping_replaces_the_old_ingredient() -> None:
    """An appending write would leave the case carrying both concepts."""
    store_matches([_match()])

    store_matches([_match(ingredients=(Concept(rxcui="161", name="acetaminophen", tty="IN"),))])

    row = DrugStringMatch.objects.get(folded_string="ZANTAC")
    assert [item.ingredient_id for item in row.ingredients.all()] == [161]


def test_the_same_string_in_both_fields_is_two_rows() -> None:
    """`prod_ai` and `drugname` are different claims about different columns."""
    store_matches([_match(source_field="prod_ai"), _match(source_field="drugname")])

    assert DrugStringMatch.objects.filter(folded_string="ZANTAC").count() == 2


def test_an_unmatched_string_is_recorded_with_no_ingredients() -> None:
    """Unmatched is a measurement. It is stored, not skipped."""
    store_matches(
        [
            _match(
                folded="QQQQ",
                cleaned="QQQQ",
                method=match.METHOD_UNMATCHED,
                rxcui=None,
                ingredients=(),
                components_matched=0,
            )
        ]
    )

    row = DrugStringMatch.objects.get(folded_string="QQQQ")
    assert row.match_method == match.METHOD_UNMATCHED
    assert row.ingredients.count() == 0
    assert row.rxcui is None
