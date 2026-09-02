"""Where the repeats go, and why the placement is not uniform.

The draw is 300 pairs in seeded random order and the annotator works a prefix of
it. Because the order is random, every prefix is itself a uniform sample of the
frame, so the sample size stops being a commitment made in advance and finishing
300 stops being a pass/fail condition.

That has one hard consequence for the repeats. Thirty of the 330 screens are a
second presentation of a pair already seen, and a consistency figure needs both
presentations inside whatever prefix was actually annotated. Repeats placed
uniformly across the schedule would put almost none of them inside the first
half, so a day that ended at screen 180 would produce no consistency figure at
all. The placement here is deliberately front-weighted:

    10 pairs complete between screens 100 and 145, one every 5 screens
    20 pairs complete between screens 159 and 330, one every 9 screens

so completions accumulate as

    prefix 100 ->  1 pair      prefix 200 -> 15 pairs
    prefix 150 -> 10 pairs     prefix 250 -> 21 pairs
    prefix 330 -> 30 pairs

Ten completed pairs is the point where the figure carries information: at 10 of
10 agreement the Wilson 95 percent lower bound is 0.72, and at 7 of 7 it is
0.65. So the figure is computable from screen 100 and meaningful from screen
150, and :data:`MEANINGFUL_PREFIX` is asserted against the schedule rather than
asserted about it.

Separation is at least 60 screens, which at any plausible pace is more than an
hour and usually a session boundary. Concealment is bounded by that plus the
sibling-string presentation; it is never claimed to be total, because the text
of a repeat is identical to its original by construction.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from signaldesk.core.errors import AnnotationError

TOTAL_UNIQUE: Final = 300
TOTAL_REPEATS: Final = 30
TOTAL_SCREENS: Final = TOTAL_UNIQUE + TOTAL_REPEATS

#: Minimum screens between a pair's two presentations.
MIN_SEPARATION: Final = 60

#: The front-weighted band: completions here are what make a short prefix usable.
EARLY_FIRST: Final = 100
EARLY_STEP: Final = 5
EARLY_COUNT: Final = 10

#: The remainder, spread evenly to the end of the schedule.
LATE_ORIGIN: Final = 150
LATE_STEP: Final = 9
LATE_COUNT: Final = TOTAL_REPEATS - EARLY_COUNT

#: Prefix length at which the consistency figure carries information.
MEANINGFUL_PREFIX: Final = 150
MEANINGFUL_PAIRS: Final = 10


def completion_positions() -> tuple[int, ...]:
    """The 1-indexed schedule positions holding a second presentation."""
    early = [EARLY_FIRST + index * EARLY_STEP for index in range(EARLY_COUNT)]
    late = [LATE_ORIGIN + (index + 1) * LATE_STEP for index in range(LATE_COUNT)]
    return tuple(early + late)


def completed_pairs_by(prefix: int) -> int:
    """How many repeat pairs have both presentations inside ``prefix`` screens."""
    return sum(1 for position in completion_positions() if position <= prefix)


@dataclass(frozen=True, slots=True)
class Placement:
    """One screen's place in the schedule.

    ``repeats_index`` is the index into the drawn pair list. Two placements
    sharing it are the two presentations of one pair, and the later one carries
    ``is_repeat``.
    """

    position: int
    pair_index: int
    is_repeat: bool


def build_schedule(pair_count: int, rng: random.Random) -> tuple[Placement, ...]:
    """Lay ``pair_count`` pairs out over the 330 screens with the repeats placed.

    The unique pairs are shuffled into the non-completion positions first, and
    only then is each completion position matched to a pair that already sits far
    enough ahead of it. Choosing the repeated pairs from where they landed rather
    than choosing them up front is what keeps the repeated set representative:
    position is independent of pair content under a uniform shuffle, so
    selecting on position selects at random with respect to what is being asked.
    """
    if pair_count != TOTAL_UNIQUE:
        message = f"schedule is defined for {TOTAL_UNIQUE} pairs, got {pair_count}"
        raise AnnotationError(message)

    completions = completion_positions()
    completion_set = set(completions)
    unique_positions = [
        position for position in range(1, TOTAL_SCREENS + 1) if position not in completion_set
    ]
    if len(unique_positions) != TOTAL_UNIQUE:
        message = (
            f"schedule arithmetic is wrong: {len(unique_positions)} unique slots for "
            f"{TOTAL_UNIQUE} pairs"
        )
        raise AnnotationError(message)

    order = list(range(pair_count))
    rng.shuffle(order)
    pair_at_position = dict(zip(unique_positions, order, strict=True))

    placements = [
        Placement(position=position, pair_index=pair_index, is_repeat=False)
        for position, pair_index in pair_at_position.items()
    ]

    repeated: set[int] = set()
    for completion in sorted(completions):
        latest_first = completion - MIN_SEPARATION
        candidates = [
            pair_at_position[position]
            for position in unique_positions
            if position <= latest_first and pair_at_position[position] not in repeated
        ]
        if not candidates:
            message = (
                f"no pair sits {MIN_SEPARATION} screens before position {completion}; "
                "the completion band and the separation are inconsistent"
            )
            raise AnnotationError(message)
        chosen = rng.choice(candidates)
        repeated.add(chosen)
        placements.append(Placement(position=completion, pair_index=chosen, is_repeat=True))

    placements.sort(key=lambda item: item.position)
    return tuple(placements)


def first_presentation_index(placements: Sequence[Placement]) -> dict[int, int]:
    """Pair index -> the schedule position of its first presentation."""
    firsts: dict[int, int] = {}
    for placement in placements:
        if not placement.is_repeat:
            firsts[placement.pair_index] = placement.position
    return firsts
