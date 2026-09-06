"""Retrieval evaluation: recall@k, MRR and nDCG over a curated gold set.

Complete and unrun. There is no gold set - `evals/golden/` is empty, P12 is
deferred with zero verdicts - so no retrieval accuracy figure exists for this
project and none should be inferred from the presence of this package. The
loader refuses rather than scoring; see `gold`.
"""

from __future__ import annotations

from signaldesk.evals.retrieval.gold import (
    GOLD_SET_FILENAME,
    GoldSet,
    Judgement,
    RetrievalGoldSetError,
    default_gold_path,
    gold_root,
    load_gold_set,
    missing_reason,
    parse_judgements,
)
from signaldesk.evals.retrieval.metrics import (
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from signaldesk.evals.retrieval.suite import (
    DEFAULT_K_VALUES,
    QueryResult,
    RetrievalReport,
    render,
    run,
)

__all__ = [
    "DEFAULT_K_VALUES",
    "GOLD_SET_FILENAME",
    "GoldSet",
    "Judgement",
    "QueryResult",
    "RetrievalGoldSetError",
    "RetrievalReport",
    "default_gold_path",
    "gold_root",
    "load_gold_set",
    "missing_reason",
    "ndcg_at_k",
    "parse_judgements",
    "recall_at_k",
    "reciprocal_rank",
    "render",
    "run",
]
