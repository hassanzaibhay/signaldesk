"""The repeat schedule, and the prefix guarantee it exists to provide."""

from __future__ import annotations

import random

import pytest

from signaldesk.core.errors import AnnotationError
from signaldesk.evals.labeledness.schedule import (
    MEANINGFUL_PAIRS,
    MEANINGFUL_PREFIX,
    MIN_SEPARATION,
    TOTAL_REPEATS,
    TOTAL_SCREENS,
    TOTAL_UNIQUE,
    build_schedule,
    completed_pairs_by,
    completion_positions,
    first_presentation_index,
)

pytestmark = pytest.mark.unit


class TestTheScheduleIsPrefixValid:
    """The annotator works a prefix, so the repeats have to work in a prefix.

    A uniform placement of 30 repeats over 330 screens puts almost none of them
    inside the first half, and a day that stops at screen 180 then produces no
    consistency figure at all. These tests pin the front-weighting that prevents
    that, at the specific prefix lengths the guideline quotes.
    """

    def test_ten_pairs_are_complete_by_the_meaningful_prefix(self) -> None:
        assert completed_pairs_by(MEANINGFUL_PREFIX) == MEANINGFUL_PAIRS

    @pytest.mark.parametrize(
        ("prefix", "expected"),
        [(99, 0), (100, 1), (150, 10), (200, 15), (250, 21), (TOTAL_SCREENS, TOTAL_REPEATS)],
    )
    def test_completions_accumulate_as_published(self, prefix: int, expected: int) -> None:
        assert completed_pairs_by(prefix) == expected

    def test_completions_are_never_all_in_the_back_half(self) -> None:
        """The failure the front-weighting exists to prevent, stated directly."""
        assert completed_pairs_by(TOTAL_SCREENS // 2) >= MEANINGFUL_PAIRS

    def test_every_completion_position_is_distinct_and_in_range(self) -> None:
        positions = completion_positions()
        assert len(positions) == TOTAL_REPEATS
        assert len(set(positions)) == TOTAL_REPEATS
        assert min(positions) >= 1
        assert max(positions) <= TOTAL_SCREENS


class TestTheBuiltSchedule:
    def test_it_places_every_pair_once_and_every_repeat_once(self) -> None:
        placements = build_schedule(TOTAL_UNIQUE, random.Random(11))
        assert len(placements) == TOTAL_SCREENS
        assert [item.position for item in placements] == list(range(1, TOTAL_SCREENS + 1))
        firsts = [item for item in placements if not item.is_repeat]
        repeats = [item for item in placements if item.is_repeat]
        assert len(firsts) == TOTAL_UNIQUE
        assert len({item.pair_index for item in firsts}) == TOTAL_UNIQUE
        assert len(repeats) == TOTAL_REPEATS
        assert len({item.pair_index for item in repeats}) == TOTAL_REPEATS

    def test_no_repeat_sits_closer_than_the_minimum_separation(self) -> None:
        placements = build_schedule(TOTAL_UNIQUE, random.Random(11))
        firsts = first_presentation_index(placements)
        for placement in placements:
            if placement.is_repeat:
                gap = placement.position - firsts[placement.pair_index]
                assert gap >= MIN_SEPARATION

    def test_a_repeat_never_precedes_its_own_first_presentation(self) -> None:
        placements = build_schedule(TOTAL_UNIQUE, random.Random(7))
        firsts = first_presentation_index(placements)
        for placement in placements:
            if placement.is_repeat:
                assert placement.position > firsts[placement.pair_index]

    def test_the_same_seed_produces_the_same_schedule(self) -> None:
        assert build_schedule(TOTAL_UNIQUE, random.Random(4242)) == build_schedule(
            TOTAL_UNIQUE, random.Random(4242)
        )

    def test_a_different_seed_produces_a_different_schedule(self) -> None:
        assert build_schedule(TOTAL_UNIQUE, random.Random(1)) != build_schedule(
            TOTAL_UNIQUE, random.Random(2)
        )

    @pytest.mark.parametrize("seed", list(range(25)))
    def test_the_placement_is_feasible_under_any_seed(self, seed: int) -> None:
        """The greedy match can only fail if the bands and the separation disagree.

        Twenty-five seeds rather than one, because a shortage would show up as a
        seed-dependent failure and a single-seed test would find it only by luck.
        """
        placements = build_schedule(TOTAL_UNIQUE, random.Random(seed))
        firsts = first_presentation_index(placements)
        assert all(
            placement.position - firsts[placement.pair_index] >= MIN_SEPARATION
            for placement in placements
            if placement.is_repeat
        )

    def test_it_refuses_a_pair_count_it_was_not_defined_for(self) -> None:
        with pytest.raises(AnnotationError, match="schedule is defined for"):
            build_schedule(299, random.Random(0))
