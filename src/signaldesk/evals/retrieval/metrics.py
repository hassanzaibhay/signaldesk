"""Ranking metrics, as pure functions on rank lists.

No I/O, no models, no database. They take a ranking and a set of relevant
identifiers and return a number, which means they can be checked against tables
computed by hand - and are, in the unit suite - rather than against whatever the
retriever happened to produce.

Three metrics, because they disagree usefully:

* **recall@k** asks whether the relevant chunks are in the window a reader will
  actually look at. It ignores order inside that window.
* **MRR** asks how far down the first relevant chunk is. It ignores everything
  after that one.
* **nDCG@k** asks about the whole window, discounting by position, and is the
  only one of the three that notices a second relevant chunk moving from rank 8
  to rank 2.

All three are undefined for a query with no relevant chunks. Rather than
returning zero - which averages in as if the retriever had failed - the loader
refuses such a judgement outright, so nothing here has to decide what it means.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def _seen_in_order(ranked: Sequence[str]) -> list[str]:
    """``ranked`` with later duplicates removed.

    A retriever returning the same chunk twice would otherwise let one relevant
    result count for two, inflating recall above what it found.
    """
    seen: set[str] = set()
    unique = []
    for item in ranked:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def recall_at_k(ranked: Sequence[str], relevant: frozenset[str], k: int) -> float:
    """Share of the relevant chunks appearing in the first ``k`` results.

    Denominator is the number of relevant chunks, not ``k``. A query with three
    relevant chunks of which two are retrieved scores 2/3 whatever ``k`` is,
    which is what makes the number comparable across queries.
    """
    if k < 1:
        message = f"k must be at least 1, got {k}"
        raise ValueError(message)
    if not relevant:
        message = "recall is undefined for a query with no relevant chunks"
        raise ValueError(message)
    window = _seen_in_order(ranked)[:k]
    return len(relevant.intersection(window)) / len(relevant)


def reciprocal_rank(ranked: Sequence[str], relevant: frozenset[str]) -> float:
    """One over the one-based rank of the first relevant chunk, or zero.

    Zero when nothing relevant appears anywhere in the ranking, which is a
    genuine result rather than an undefined one: the retriever returned a list
    and none of it was relevant.
    """
    if not relevant:
        message = "reciprocal rank is undefined for a query with no relevant chunks"
        raise ValueError(message)
    for position, chunk in enumerate(_seen_in_order(ranked), start=1):
        if chunk in relevant:
            return 1.0 / position
    return 0.0


def dcg_at_k(ranked: Sequence[str], relevant: frozenset[str], k: int) -> float:
    """Discounted cumulative gain over binary relevance."""
    window = _seen_in_order(ranked)[:k]
    return sum(
        1.0 / math.log2(position + 1)
        for position, chunk in enumerate(window, start=1)
        if chunk in relevant
    )


def ndcg_at_k(ranked: Sequence[str], relevant: frozenset[str], k: int) -> float:
    """nDCG@k over binary relevance.

    The ideal ranking puts every relevant chunk first, so the ideal DCG depends
    on how many relevant chunks exist and on ``k`` - with five relevant chunks
    and k of 3 a perfect retriever still scores 1.0, because three is all the
    window can hold. Normalising against a fixed ideal instead would report a
    perfect retriever as imperfect for a reason that has nothing to do with it.
    """
    if k < 1:
        message = f"k must be at least 1, got {k}"
        raise ValueError(message)
    if not relevant:
        message = "nDCG is undefined for a query with no relevant chunks"
        raise ValueError(message)
    ideal = sum(1.0 / math.log2(position + 1) for position in range(1, min(len(relevant), k) + 1))
    return dcg_at_k(ranked, relevant, k) / ideal


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean, or zero over nothing.

    Zero over an empty sequence is safe here only because the caller cannot
    reach it: a report over zero queries is refused before any metric is
    averaged. It is written rather than left to raise so that the refusal stays
    the single place that decides what an empty evaluation means.
    """
    return sum(values) / len(values) if values else 0.0
