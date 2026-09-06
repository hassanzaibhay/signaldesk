"""Fusing the two retrievers, reranking what they agree on, and returning few.

The shape, with every number coming from the settings object rather than from
this file:

    dense_top_k   50  ->  |
                          |  RRF at rrf_k = 60  ->  at most 100 unique
    sparse_top_k  50  ->  |
                                        -> cross-encoder -> rerank_top_k = 8

Reciprocal rank fusion reads ranks and ignores scores, which is the property
that makes it usable here. A BM25 score and a cosine distance are not on one
scale and no principled constant converts them; their orderings are comparable
and their magnitudes are not. RRF is the standard way to combine rankers whose
scores cannot be compared, and its one parameter damps the contribution of deep
ranks so that a result appearing at rank 40 in both lists does not outweigh one
appearing at rank 1 in either.

The candidate pool entering the cross-encoder is at most ``dense_top_k +
sparse_top_k``, less whatever the two retrievers agreed on. Every pair costs a
forward pass, so on the four-core container this is the expensive step of a
query by a wide margin. Nothing interactive consumes it today.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from signaldesk.core.config import Settings, get_settings
from signaldesk.core.logging import get_logger
from signaldesk.rag.embed import CrossEncoder, Encoder, embed_texts
from signaldesk.rag.index import dense, sparse
from signaldesk.web.documents.models import EMBEDDING_DIMENSIONS, LabelChunk

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Fused:
    """One candidate, and where each retriever placed it."""

    chunk_id: int
    score: float
    #: One-based rank in that retriever's list, or None if it never appeared.
    #: Kept because "found by both" and "found by one" is the most useful thing
    #: to know about a candidate when a result looks wrong.
    dense_rank: int | None
    sparse_rank: int | None


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A chunk that survived to the end, with its full provenance."""

    chunk_id: int
    text: str
    section_code: str
    rrf_score: float
    dense_rank: int | None
    sparse_rank: int | None
    #: None when no cross-encoder was applied, which is a different state from
    #: a score of zero and is kept distinguishable.
    rerank_score: float | None


def reciprocal_rank_fusion(
    dense_ids: Sequence[int], sparse_ids: Sequence[int], *, rrf_k: int
) -> tuple[Fused, ...]:
    """Fuse two ranked id lists into one, best first.

    Ties break on chunk id ascending, so a fused list is deterministic. Two
    candidates found at the same rank by the same single retriever genuinely
    have equal evidence, and leaving their order to dictionary insertion would
    make an evaluation result depend on which retriever ran first.
    """
    if rrf_k < 1:
        message = f"rrf_k must be at least 1, got {rrf_k}"
        raise ValueError(message)

    ranks: dict[int, list[int | None]] = {}
    for position, chunk_id in enumerate(dense_ids, start=1):
        ranks.setdefault(chunk_id, [None, None])[0] = position
    for position, chunk_id in enumerate(sparse_ids, start=1):
        ranks.setdefault(chunk_id, [None, None])[1] = position

    fused = [
        Fused(
            chunk_id=chunk_id,
            score=sum(1.0 / (rrf_k + rank) for rank in pair if rank is not None),
            dense_rank=pair[0],
            sparse_rank=pair[1],
        )
        for chunk_id, pair in ranks.items()
    ]
    fused.sort(key=lambda candidate: (-candidate.score, candidate.chunk_id))
    return tuple(fused)


def rerank(
    query: str,
    candidates: Sequence[Fused],
    texts: dict[int, str],
    cross_encoder: CrossEncoder,
    *,
    top_k: int,
) -> tuple[tuple[Fused, float], ...]:
    """Rescore ``candidates`` jointly against ``query`` and keep the best.

    Ties break on the fused order, which is already deterministic, so an
    unhelpful cross-encoder that returns one score for everything degrades to
    the fusion result rather than to an arbitrary one.
    """
    if not candidates:
        return ()
    ordered = list(candidates)
    scores = cross_encoder.score(query, [texts[candidate.chunk_id] for candidate in ordered])
    if len(scores) != len(ordered):
        message = (
            f"the cross-encoder returned {len(scores)} scores for {len(ordered)} "
            "candidates; scores and candidates correspond by position"
        )
        raise ValueError(message)
    paired = list(zip(ordered, (float(score) for score in scores), strict=True))
    paired.sort(key=lambda item: (-item[1], -item[0].score, item[0].chunk_id))
    return tuple(paired[:top_k])


def retrieve(
    query: str,
    *,
    encoder: Encoder,
    cross_encoder: CrossEncoder | None,
    sparse_index: sparse.SparseIndex,
    embedding_model: str,
    document_ids: Sequence[int] | None = None,
    settings: Settings | None = None,
) -> tuple[RetrievedChunk, ...]:
    """Run the whole pipeline for one query.

    ``cross_encoder`` may be None, which runs dense, sparse and fusion and skips
    the rerank. That is a real configuration - it is what an ablation of the
    reranker is - and not a fallback for a missing model.
    """
    settings = settings or get_settings()

    query_vector = embed_texts(
        encoder, [query], expected_dimensions=EMBEDDING_DIMENSIONS, batch_size=1
    )
    dense_hits = dense.search(
        query_vector[0],
        settings.dense_top_k,
        model=embedding_model,
        document_ids=document_ids,
        settings=settings,
    )
    sparse_hits = sparse_index.search(query, settings.sparse_top_k)

    fused = reciprocal_rank_fusion(
        [hit.chunk_id for hit in dense_hits],
        [hit.chunk_id for hit in sparse_hits],
        rrf_k=settings.rrf_k,
    )
    if not fused:
        return ()

    rows = LabelChunk.objects.filter(
        id__in=[candidate.chunk_id for candidate in fused]
    ).values_list("id", "text", "section_code")
    texts = {chunk_id: text for chunk_id, text, _ in rows}
    codes = {chunk_id: code for chunk_id, _, code in rows}
    # A candidate whose row is gone is dropped rather than carried with an empty
    # text. It can only happen if the corpus changed under a stale index, and a
    # blank chunk in a result list looks like a retrieval defect rather than a
    # stale index, which is the wrong thing to be debugging.
    present = tuple(candidate for candidate in fused if candidate.chunk_id in texts)
    if len(present) != len(fused):
        log.warning(
            "rag.retrieve.stale_index",
            missing=len(fused) - len(present),
            hint="the sparse or dense index names chunks that no longer exist; rebuild",
        )

    if cross_encoder is None:
        return tuple(
            RetrievedChunk(
                chunk_id=candidate.chunk_id,
                text=texts[candidate.chunk_id],
                section_code=codes[candidate.chunk_id],
                rrf_score=candidate.score,
                dense_rank=candidate.dense_rank,
                sparse_rank=candidate.sparse_rank,
                rerank_score=None,
            )
            for candidate in present[: settings.rerank_top_k]
        )

    reranked = rerank(query, present, texts, cross_encoder, top_k=settings.rerank_top_k)
    return tuple(
        RetrievedChunk(
            chunk_id=candidate.chunk_id,
            text=texts[candidate.chunk_id],
            section_code=codes[candidate.chunk_id],
            rrf_score=candidate.score,
            dense_rank=candidate.dense_rank,
            sparse_rank=candidate.sparse_rank,
            rerank_score=score,
        )
        for candidate, score in reranked
    )
