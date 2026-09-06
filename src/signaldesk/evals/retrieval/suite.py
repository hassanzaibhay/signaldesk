"""Running the retrieval evaluation, and refusing to when there is nothing to run.

The suite is a function from a gold set and a retriever to a report. The
retriever is passed in as a callable from query text to a ranked list of chunk
hashes, which keeps this module independent of how retrieval is assembled: the
integration test hands it the real pipeline, the unit tests hand it a list, and
neither needs the other's dependencies.

Nothing here writes to `evals/history/`. The report has an `as_dict` for a
caller that wants to persist one, and no caller does yet, because there is no
gold set to produce a report from. That ordering is deliberate. The first
retrieval number this project publishes should be the output of a run over
judgements a person made, and the way to guarantee that is for the code that
could write a file to not exist until then.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from signaldesk.core.logging import get_logger
from signaldesk.evals.retrieval import metrics
from signaldesk.evals.retrieval.gold import GoldSet, Judgement, RetrievalGoldSetError

log = get_logger(__name__)

#: Cutoffs reported for recall and nDCG. 1 is "did it lead with a right
#: answer", 5 and 10 bracket what a reader scans, and 8 is `rerank_top_k`, the
#: size of the list the pipeline actually returns.
DEFAULT_K_VALUES: tuple[int, ...] = (1, 5, 8, 10)

#: A retriever, as this suite needs it: query text to ranked chunk hashes.
Retriever = Callable[[str], Sequence[str]]


@dataclass(frozen=True, slots=True)
class QueryResult:
    """What one query scored, kept so a report can be read per query.

    A mean over 300 queries hides the shape of the distribution, and the useful
    question about a retrieval result is almost always which queries failed
    rather than what the average was.
    """

    query_id: str
    retrieved: int
    relevant: int
    recall_at: dict[int, float]
    reciprocal_rank: float
    ndcg_at: dict[int, float]


@dataclass(frozen=True, slots=True)
class RetrievalReport:
    """Aggregate metrics, and the per-query results behind them."""

    queries: int
    k_values: tuple[int, ...]
    recall_at: dict[int, float]
    mrr: float
    ndcg_at: dict[int, float]
    per_query: tuple[QueryResult, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact": "retrieval",
            "queries": self.queries,
            "k_values": list(self.k_values),
            "recall_at": {str(k): round(value, 6) for k, value in self.recall_at.items()},
            "mrr": round(self.mrr, 6),
            "ndcg_at": {str(k): round(value, 6) for k, value in self.ndcg_at.items()},
            "per_query": [
                {
                    "query_id": result.query_id,
                    "retrieved": result.retrieved,
                    "relevant": result.relevant,
                    "recall_at": {str(k): round(v, 6) for k, v in result.recall_at.items()},
                    "reciprocal_rank": round(result.reciprocal_rank, 6),
                    "ndcg_at": {str(k): round(v, 6) for k, v in result.ndcg_at.items()},
                }
                for result in self.per_query
            ],
        }


def score_query(
    judgement: Judgement, ranked: Sequence[str], k_values: Sequence[int]
) -> QueryResult:
    """Metrics for one query against one ranking."""
    return QueryResult(
        query_id=judgement.query_id,
        retrieved=len(ranked),
        relevant=len(judgement.relevant),
        recall_at={k: metrics.recall_at_k(ranked, judgement.relevant, k) for k in k_values},
        reciprocal_rank=metrics.reciprocal_rank(ranked, judgement.relevant),
        ndcg_at={k: metrics.ndcg_at_k(ranked, judgement.relevant, k) for k in k_values},
    )


def run(
    gold: GoldSet,
    retriever: Retriever,
    *,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> RetrievalReport:
    """Score ``retriever`` over every judgement in ``gold``.

    Refuses an empty gold set rather than reporting a mean over nothing. The
    loader already refuses one, so reaching this needs a `GoldSet` built by
    hand; it is checked anyway because a metric of 0.0 over zero queries is
    indistinguishable from a retriever that found nothing, and one of those is a
    result while the other is a bug.
    """
    if not gold.judgements:
        message = (
            "refusing to score an empty gold set. A mean over zero queries is "
            "not a retrieval metric."
        )
        raise RetrievalGoldSetError(message)
    if not k_values:
        message = "at least one k is needed to report recall and nDCG"
        raise ValueError(message)

    cutoffs = tuple(sorted(set(k_values)))
    results = tuple(
        score_query(judgement, list(retriever(judgement.query)), cutoffs)
        for judgement in gold.judgements
    )

    log.info("evals.retrieval.scored", queries=len(results), k_values=list(cutoffs))
    return RetrievalReport(
        queries=len(results),
        k_values=cutoffs,
        recall_at={k: metrics.mean([result.recall_at[k] for result in results]) for k in cutoffs},
        mrr=metrics.mean([result.reciprocal_rank for result in results]),
        ndcg_at={k: metrics.mean([result.ndcg_at[k] for result in results]) for k in cutoffs},
        per_query=results,
    )


def render(report: RetrievalReport) -> str:
    """The report as a short block of text, for a terminal."""
    lines = [f"queries: {report.queries}", f"MRR: {report.mrr:.4f}"]
    lines.extend(f"recall@{k}: {report.recall_at[k]:.4f}" for k in report.k_values)
    lines.extend(f"nDCG@{k}: {report.ndcg_at[k]:.4f}" for k in report.k_values)
    return "\n".join(lines)
