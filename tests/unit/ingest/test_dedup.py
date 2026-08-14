"""Deduplication decisions, on hand-built cases.

The decision function is isolated from the blocking machinery precisely so it can
be tested like this: every clause of the rule gets a case that fails only that
clause, because a rule that is only ever tested on records that match is a rule
nobody has checked.
"""

from __future__ import annotations

import pytest

from signaldesk.core.errors import IngestError
from signaldesk.ingest.faers.dedup import (
    DRUG_THRESHOLD,
    REACTION_THRESHOLD,
    BlockRecord,
    DedupStats,
    choose_survivor,
    is_duplicate,
    jaccard,
    prefix_length,
    resolve_chains,
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


def _record(primaryid: int, fda_dt: str, **overrides: object) -> BlockRecord:
    fields: dict[str, object] = {
        "index": primaryid,
        "primaryid": primaryid,
        "caseid": primaryid // 10,
        "quarter": "2014Q2",
        "sex": "F",
        "country": "US",
        "age": 40.0,
        "fda_dt": fda_dt,
        "drugs": frozenset({"aspirin"}),
        "reactions": frozenset({"nausea"}),
    }
    fields.update(overrides)
    return BlockRecord(**fields)  # type: ignore[arg-type]


def test_the_later_report_survives() -> None:
    early, late = _record(1, "2014-05-01"), _record(2, "2014-06-01")
    assert choose_survivor(early, late) == (late, early)


def test_the_tie_break_is_a_total_order() -> None:
    """The same pair in either argument position yields the same survivor.

    ``>=`` holds in both directions on equal dates, so with it the survivor was
    decided by which side of the call a record landed on. That is scoring order,
    not data, and it let the flag graph close 17 loops in which every member was
    flagged and 38 cases survived nowhere.
    """
    left, right = _record(101, "2014-05-08"), _record(202, "2014-05-08")

    assert choose_survivor(left, right) == choose_survivor(right, left)
    assert choose_survivor(left, right)[0].primaryid == 202


def test_population_excludes_what_the_rule_cannot_reach() -> None:
    """A record that cannot reach the numerator does not sit in the denominator."""
    stats = DedupStats(
        records_considered=600,
        records_excluded_single_member_block=50,
        records_excluded_null_key=400,
        records_excluded_superseded=300,
        records_excluded_empty_drug_set=10,
    )

    # Alone in its block is a legitimate zero: the rule applied and found nothing.
    assert stats.population == 650
    assert stats.records_evaluated == 1360
    assert stats.excluded_share == pytest.approx(400 / 1360)


def test_resolve_chains_reports_a_terminating_chain_as_sound() -> None:
    report = resolve_chains([(1, 2), (2, 3)])

    assert report.is_sound
    assert report.closed_components == 0
    assert report.cycles == 0
    assert report.resolved_to_unflagged == 2
    assert report.components == 1


def test_resolve_chains_detects_a_three_cycle() -> None:
    """The shape P02 found: every member flagged, so no case survives it."""
    report = resolve_chains([(1, 2), (2, 3), (3, 1), (4, 5)])

    assert not report.is_sound
    assert report.cycles == 1
    assert report.cycle_lengths == (3,)
    assert report.records_in_cycles == 3
    assert report.closed_components == 1
    assert report.records_in_closed_components == 3
    assert report.closed_component_members == ((1, 2, 3),)
    # The unrelated pair still resolves.
    assert report.resolved_to_unflagged == 1
    assert report.resolved_into_cycle == 3


def test_resolve_chains_counts_records_hanging_off_a_cycle() -> None:
    report = resolve_chains([(1, 2), (2, 3), (3, 1), (4, 1)])

    assert report.records_in_cycles == 3
    assert report.records_in_closed_components == 4
    assert report.resolved_into_cycle == 4


def test_a_record_flagged_twice_raises_rather_than_resolving_arbitrarily() -> None:
    with pytest.raises(IngestError, match="flagged twice"):
        resolve_chains([(1, 2), (1, 3)])
