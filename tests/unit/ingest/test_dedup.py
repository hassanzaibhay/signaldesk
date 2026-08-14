"""Deduplication decisions, on hand-built cases.

The decision function is isolated from the blocking machinery precisely so it can
be tested like this: every clause of the rule gets a case that fails only that
clause, because a rule that is only ever tested on records that match is a rule
nobody has checked.
"""

from __future__ import annotations

import pytest

from signaldesk.ingest.faers.dedup import (
    DRUG_THRESHOLD,
    REACTION_THRESHOLD,
    DedupStats,
    is_duplicate,
    jaccard,
    prefix_length,
)

pytestmark = pytest.mark.unit


def test_jaccard_of_identical_sets_is_one() -> None:
    assert jaccard(frozenset({"a", "b"}), frozenset({"a", "b"})) == 1.0


def test_jaccard_of_disjoint_sets_is_zero() -> None:
    assert jaccard(frozenset({"a"}), frozenset({"b"})) == 0.0


def test_jaccard_is_intersection_over_union() -> None:
    assert jaccard(frozenset({"a", "b", "c"}), frozenset({"b", "c", "d"})) == pytest.approx(0.5)


def test_empty_sets_are_not_similar() -> None:
    # No evidence is not evidence of sameness.
    assert jaccard(frozenset(), frozenset()) == 0.0


def _decide(**overrides: object) -> bool:
    arguments: dict[str, object] = {
        "drug_similarity": 1.0,
        "reaction_similarity": 1.0,
        "same_sex": True,
        "age_difference": 0.0,
        "same_country": True,
    }
    arguments.update(overrides)
    return is_duplicate(**arguments)  # type: ignore[arg-type]


def test_a_matching_pair_is_a_duplicate() -> None:
    assert _decide() is True


def test_thresholds_are_inclusive() -> None:
    assert _decide(drug_similarity=DRUG_THRESHOLD, reaction_similarity=REACTION_THRESHOLD) is True


def test_near_miss_on_drugs_is_not_merged() -> None:
    # The real near-miss pair carried by the fixture: reactions match exactly,
    # drug sets do not. Two patients with the same common event, not one case.
    assert _decide(drug_similarity=0.64, reaction_similarity=1.0) is False


def test_near_miss_on_reactions_is_not_merged() -> None:
    assert _decide(drug_similarity=1.0, reaction_similarity=0.7) is False


def test_age_difference_beyond_a_year_is_not_merged() -> None:
    assert _decide(age_difference=2.0) is False


def test_age_difference_of_exactly_a_year_is_allowed() -> None:
    assert _decide(age_difference=1.0) is True


def test_unknown_age_difference_is_not_merged() -> None:
    assert _decide(age_difference=None) is False


def test_different_sex_or_country_is_not_merged() -> None:
    assert _decide(same_sex=False) is False
    assert _decide(same_country=False) is False


@pytest.mark.parametrize(
    ("size", "expected"),
    [(1, 1), (5, 2), (10, 3), (20, 5)],
)
def test_prefix_length_grows_with_set_size(size: int, expected: int) -> None:
    # For threshold 0.8, a set of n needs n - ceil(0.8n) + 1 indexed tokens for
    # the filter to stay exact.
    assert prefix_length(size, 0.8) == expected


def test_cross_quarter_share_is_zero_without_duplicates() -> None:
    assert DedupStats().cross_quarter_share == 0.0


def test_cross_quarter_share_counts_pairs_spanning_quarters() -> None:
    stats = DedupStats(duplicate_pairs=4, cross_quarter_pairs=3)
    assert stats.cross_quarter_share == pytest.approx(0.75)


def test_excluded_share_is_against_everything_considered() -> None:
    stats = DedupStats(records_considered=60, records_excluded_null_key=40)
    assert stats.excluded_share == pytest.approx(0.4)
