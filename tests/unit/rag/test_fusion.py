"""Reciprocal rank fusion and cross-encoder reranking.

Both are pure functions on rank lists. The RRF scores below are worked out by
hand, because the property that matters is not "it produces an order" but "it
produces this order", and the constant is what decides how much a deep rank in
one list is worth against a shallow rank in the other.
"""

from __future__ import annotations

import pytest
from tests.conftest import OverlapCrossEncoder

from signaldesk.rag.retrieve import Fused, reciprocal_rank_fusion, rerank

pytestmark = pytest.mark.unit


class TestFusion:
    def test_scores_are_the_sum_of_reciprocal_ranks(self) -> None:
        # Chunk 1: dense rank 1 and sparse rank 2 -> 1/61 + 1/62 = 0.0325222
        # Chunk 3: dense rank 3 and sparse rank 1 -> 1/63 + 1/61 = 0.0322661
        # Chunk 2: dense rank 2 only              -> 1/62       = 0.0161290
        # Chunk 4: sparse rank 3 only             -> 1/63       = 0.0158730
        fused = reciprocal_rank_fusion([1, 2, 3], [3, 1, 4], rrf_k=60)

        assert [candidate.chunk_id for candidate in fused] == [1, 3, 2, 4]
        assert fused[0].score == pytest.approx(1 / 61 + 1 / 62)
        assert fused[1].score == pytest.approx(1 / 63 + 1 / 61)
        assert fused[2].score == pytest.approx(1 / 62)

    def test_agreement_beats_a_single_better_rank(self) -> None:
        """The entire reason for fusing rather than picking one retriever."""
        fused = reciprocal_rank_fusion([2, 1], [1], rrf_k=60)

        # Chunk 1: rank 2 dense plus rank 1 sparse. Chunk 2: rank 1 dense only.
        assert fused[0].chunk_id == 1

    def test_both_ranks_are_recorded(self) -> None:
        """ "Found by both" against "found by one" is the useful diagnostic."""
        fused = {
            candidate.chunk_id: candidate
            for candidate in reciprocal_rank_fusion([1, 2], [2, 3], rrf_k=60)
        }

        assert (fused[1].dense_rank, fused[1].sparse_rank) == (1, None)
        assert (fused[2].dense_rank, fused[2].sparse_rank) == (2, 1)
        assert (fused[3].dense_rank, fused[3].sparse_rank) == (None, 2)

    def test_ties_break_on_chunk_id_so_the_order_is_deterministic(self) -> None:
        """Otherwise an evaluation result depends on dictionary insertion order."""
        forward = reciprocal_rank_fusion([7, 3], [], rrf_k=60)
        assert [candidate.chunk_id for candidate in forward] == [7, 3]

        tied = reciprocal_rank_fusion([], [9, 2], rrf_k=60)
        assert [candidate.chunk_id for candidate in tied] == [9, 2]

    def test_one_empty_ranking_degrades_to_the_other(self) -> None:
        """A sparse-only pipeline is a real configuration, not a broken one."""
        fused = reciprocal_rank_fusion([], [5, 6], rrf_k=60)

        assert [candidate.chunk_id for candidate in fused] == [5, 6]
        assert fused[0].dense_rank is None

    def test_two_empty_rankings_fuse_to_nothing(self) -> None:
        assert reciprocal_rank_fusion([], [], rrf_k=60) == ()

    def test_a_larger_constant_flattens_the_difference_between_ranks(self) -> None:
        """What rrf_k controls, asserted rather than described."""
        tight = reciprocal_rank_fusion([1, 2], [], rrf_k=1)
        loose = reciprocal_rank_fusion([1, 2], [], rrf_k=1000)

        assert tight[0].score / tight[1].score > loose[0].score / loose[1].score

    def test_a_constant_below_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            reciprocal_rank_fusion([1], [], rrf_k=0)


class TestRerank:
    def _candidates(self) -> tuple[Fused, ...]:
        return reciprocal_rank_fusion([1, 2, 3], [], rrf_k=60)

    def test_the_cross_encoder_reorders_the_fused_list(self) -> None:
        """Chunk 3 is last after fusion and first after reranking."""
        texts = {1: "hepatic failure", 2: "rash", 3: "lactic acidosis reported"}

        ranked = rerank(
            "lactic acidosis", self._candidates(), texts, OverlapCrossEncoder(), top_k=3
        )

        assert [candidate.chunk_id for candidate, _ in ranked] == [3, 1, 2]

    def test_the_cutoff_is_applied(self) -> None:
        texts = {1: "hepatic failure", 2: "rash", 3: "lactic acidosis reported"}

        ranked = rerank(
            "lactic acidosis", self._candidates(), texts, OverlapCrossEncoder(), top_k=1
        )

        assert len(ranked) == 1

    def test_an_uninformative_cross_encoder_degrades_to_the_fused_order(self) -> None:
        """A scorer that says nothing must not scramble what fusion decided."""
        texts = {1: "alpha", 2: "beta", 3: "gamma"}

        ranked = rerank(
            "nothing matches", self._candidates(), texts, OverlapCrossEncoder(), top_k=3
        )

        assert [candidate.chunk_id for candidate, _ in ranked] == [1, 2, 3]

    def test_no_candidates_reranks_to_nothing(self) -> None:
        assert rerank("q", (), {}, OverlapCrossEncoder(), top_k=8) == ()

    def test_a_score_count_that_does_not_match_the_candidates_is_refused(self) -> None:
        """Scores and candidates correspond by position; a mismatch mislabels all of them."""

        class Short(OverlapCrossEncoder):
            def score(self, query, texts):  # type: ignore[no-untyped-def]
                return super().score(query, texts)[:-1]

        texts = {1: "a", 2: "b", 3: "c"}

        with pytest.raises(ValueError, match="correspond by position"):
            rerank("q", self._candidates(), texts, Short(), top_k=3)
