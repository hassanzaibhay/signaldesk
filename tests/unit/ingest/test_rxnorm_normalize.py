"""Drug string cleaning, on strings taken from the corpus.

Every case here is either a real string from the loaded data or a minimal
variation on one. The cases that must *not* change matter more than the ones that
must: RxNav's approximate matcher returns a plausible ingredient for almost any
input, so a transformation that guesses does not fail loudly, it produces a
confident match to the wrong concept.
"""

from __future__ import annotations

import pytest

from signaldesk.ingest.rxnorm.normalize import (
    clean,
    fold,
    split_components,
    strip_dose_and_form,
    strip_trailing_punctuation,
    strip_uninformative_brackets,
)

pytestmark = pytest.mark.unit


def test_folding_is_the_form_the_corpus_was_counted_in() -> None:
    assert fold("  humira   40  ") == "HUMIRA 40"
    assert fold("Zantac") == "ZANTAC"


def test_a_trailing_full_stop_is_dropped() -> None:
    # 423,722 rows say RANITIDINE. against 206,790 that say RANITIDINE.
    assert strip_trailing_punctuation("RANITIDINE.") == "RANITIDINE"


def test_punctuation_inside_a_name_is_left_alone() -> None:
    assert strip_trailing_punctuation("PFIZER-BIONTECH COVID-19 VACCINE") == (
        "PFIZER-BIONTECH COVID-19 VACCINE"
    )
    assert strip_trailing_punctuation("VITAMIN B-12") == "VITAMIN B-12"


def test_a_string_of_only_punctuation_survives_rather_than_becoming_empty() -> None:
    assert strip_trailing_punctuation("...") == "..."


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("RANITIDINE HYDROCHLORIDE (852)", "RANITIDINE HYDROCHLORIDE"),
        ("RITUXIMAB (UNKNOWN)", "RITUXIMAB"),
        ("OXYCODONE HCL TABLETS (RHODES 91?490)", "OXYCODONE HCL TABLETS"),
    ],
)
def test_a_bracketed_code_or_absence_is_dropped(raw: str, expected: str) -> None:
    assert strip_uninformative_brackets(raw) == expected


def test_a_bracketed_ingredient_is_kept() -> None:
    """The brackets carry the information here, so removing them loses it."""
    assert strip_uninformative_brackets("ASCORBIC ACID (VITAMIN C)") == (
        "ASCORBIC ACID (VITAMIN C)"
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("SOLIRIS 300MG", "SOLIRIS"),
        ("CAPECITABINE TAB 500MG", "CAPECITABINE"),
        ("HUMIRA 40 MG/0.8 ML PEN", "HUMIRA"),
        ("ASPIRIN 81 MG TABLET", "ASPIRIN"),
        ("OXYCODONE HCL TABLETS", "OXYCODONE HCL"),
    ],
)
def test_a_dose_or_form_on_the_tail_is_removed(raw: str, expected: str) -> None:
    assert strip_dose_and_form(raw) == expected


def test_a_number_that_is_part_of_the_name_is_kept() -> None:
    """`COVID-19` and `B12` are names, not doses."""
    assert strip_dose_and_form("COVID-19 VACCINE") == "COVID-19 VACCINE"
    assert strip_dose_and_form("VITAMIN D3") == "VITAMIN D3"


def test_stripping_never_empties_the_string() -> None:
    assert strip_dose_and_form("500MG") == "500MG"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "HYDROCODONE BITARTRATE AND ACETAMINOPHEN",
            ("HYDROCODONE BITARTRATE", "ACETAMINOPHEN"),
        ),
        ("HYDROCODONE/ACETAMINOPHEN", ("HYDROCODONE", "ACETAMINOPHEN")),
        ("SULFAMETHOXAZOLE AND TRIMETHOPRIM", ("SULFAMETHOXAZOLE", "TRIMETHOPRIM")),
    ],
)
def test_a_combination_is_split_into_every_component(raw: str, expected: tuple[str, ...]) -> None:
    assert split_components(raw) == expected


def test_a_dose_that_looks_like_a_combination_is_not_split() -> None:
    """`40 MG/0.8 ML` contains a slash and is one drug.

    The dose is removed before splitting, so by the time this runs the slash is
    gone. Asserted end to end because the ordering is the thing that makes it
    work.
    """
    result = clean("HUMIRA 40 MG/0.8 ML PEN")

    assert result.cleaned == "HUMIRA"
    assert result.components == ("HUMIRA",)
    assert not result.is_combination


def test_a_split_that_would_produce_debris_is_refused() -> None:
    assert split_components("VITAMIN B1/B6") == ("VITAMIN B1/B6",)
    assert split_components("DRUG/5") == ("DRUG/5",)


def test_the_transformation_trail_is_recorded() -> None:
    result = clean("  ranitidine hydrochloride (852) 150mg tablet.  ")

    assert result.folded == "RANITIDINE HYDROCHLORIDE (852) 150MG TABLET."
    assert result.cleaned == "RANITIDINE HYDROCHLORIDE"
    # The bracketed code only reaches the tail once the dose and form are gone,
    # which is why the strips run in rounds and the trail shows the real order.
    assert result.steps == ("trailing_punctuation", "dose_and_form", "uninformative_brackets")


def test_a_string_needing_nothing_records_nothing() -> None:
    result = clean("METHOTREXATE")

    assert result.cleaned == "METHOTREXATE"
    assert result.steps == ()
    assert result.components == ("METHOTREXATE",)


def test_a_string_that_is_not_a_drug_is_left_exactly_as_it_is() -> None:
    """Cleaning does not decide what is a drug. It only removes formatting."""
    result = clean("UNSPECIFIED INGREDIENT")

    assert result.cleaned == "UNSPECIFIED INGREDIENT"
    assert result.steps == ()
