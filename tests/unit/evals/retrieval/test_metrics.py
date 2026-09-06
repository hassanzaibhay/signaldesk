"""Ranking metrics against values computed by hand.

Every expected number below is worked out in the test that asserts it. A metric
checked against its own implementation's output tests that the code has not
changed, not that it is right.
"""

from __future__ import annotations

import math

import pytest

from signaldesk.evals.retrieval.metrics import (
    dcg_at_k,
    mean,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)

pytestmark = pytest.mark.unit

RANKED = ["a", "b", "c", "d"]
RELEVANT = frozenset({"a", "c"})


class TestRecall:
    @pytest.mark.parametrize(
        ("k", "expected"),
        [
            # One of two relevant chunks inside the window.
            (1, 0.5),
            (2, 0.5),
            # Both of two.
            (3, 1.0),
            (4, 1.0),
        ],
    )
    def test_recall_at_k(self, k: int, expected: float) -> None:
        assert recall_at_k(RANKED, RELEVANT, k) == pytest.approx(expected)

    def test_the_denominator_is_the_relevant_count_not_k(self) -> None:
        """Otherwise the number is not comparable between queries."""
        assert recall_at_k(["a"], frozenset({"a", "b", "c", "d"}), 10) == pytest.approx(0.25)

    def test_a_repeated_result_cannot_count_twice(self) -> None:
        """A retriever returning one chunk twice has still found one chunk."""
        assert recall_at_k(["a", "a"], frozenset({"a", "c"}), 2) == pytest.approx(0.5)

    def test_no_relevant_chunks_is_undefined_rather_than_zero(self) -> None:
        """Zero would average in as a retrieval failure. It is not one."""
        with pytest.raises(ValueError, match="undefined"):
            recall_at_k(RANKED, frozenset(), 3)

    def test_k_below_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            recall_at_k(RANKED, RELEVANT, 0)


class TestReciprocalRank:
    def test_the_first_relevant_result_sets_the_score(self) -> None:
        # "a" is relevant and sits at one-based rank 2.
        assert reciprocal_rank(["b", "a", "c"], RELEVANT) == pytest.approx(0.5)

    def test_leading_with_a_relevant_result_scores_one(self) -> None:
        assert reciprocal_rank(["a", "b"], RELEVANT) == pytest.approx(1.0)

    def test_nothing_relevant_anywhere_scores_zero(self) -> None:
        """A genuine result: the retriever returned a list and it was all wrong."""
        assert reciprocal_rank(["x", "y"], RELEVANT) == 0.0

    def test_no_relevant_chunks_is_undefined(self) -> None:
        with pytest.raises(ValueError, match="undefined"):
            reciprocal_rank(RANKED, frozenset())


class TestNdcg:
    def test_dcg_discounts_by_position(self) -> None:
        # Relevant at ranks 1 and 3: 1/log2(2) + 1/log2(4) = 1.0 + 0.5.
        assert dcg_at_k(RANKED, RELEVANT, 4) == pytest.approx(1.5)

    def test_ndcg_against_a_hand_computed_ideal(self) -> None:
        # DCG  = 1/log2(2) + 1/log2(4)  = 1.0 + 0.5       = 1.5
        # IDCG = 1/log2(2) + 1/log2(3)  = 1.0 + 0.630930  = 1.630930
        expected = 1.5 / (1.0 + 1.0 / math.log2(3))

        assert ndcg_at_k(RANKED, RELEVANT, 4) == pytest.approx(expected)
        assert ndcg_at_k(RANKED, RELEVANT, 4) == pytest.approx(0.919721, abs=1e-6)

    def test_a_perfect_ranking_scores_one(self) -> None:
        assert ndcg_at_k(["a", "c", "b"], RELEVANT, 3) == pytest.approx(1.0)

    def test_a_window_smaller_than_the_relevant_set_can_still_score_one(self) -> None:
        """Two relevant chunks and a window of one: finding either is perfect.

        Normalising against a fixed ideal instead would mark a retriever down
        for the size of the window rather than for anything it did.
        """
        assert ndcg_at_k(["a", "z"], RELEVANT, 1) == pytest.approx(1.0)

    def test_ordering_inside_the_window_matters(self) -> None:
        """The property recall does not have, and the reason nDCG is reported."""
        better = ndcg_at_k(["a", "c", "x"], RELEVANT, 3)
        worse = ndcg_at_k(["x", "a", "c"], RELEVANT, 3)

        assert better > worse

    def test_no_relevant_chunks_is_undefined(self) -> None:
        with pytest.raises(ValueError, match="undefined"):
            ndcg_at_k(RANKED, frozenset(), 3)


class TestMean:
    def test_mean_of_values(self) -> None:
        assert mean([0.0, 0.5, 1.0]) == pytest.approx(0.5)

    def test_mean_of_nothing_is_zero(self) -> None:
        """Reachable only if the empty-gold-set refusal is removed."""
        assert mean([]) == 0.0
