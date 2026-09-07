"""The record one index build leaves behind.

Written to ``evals/history/`` so that any number describing this index traces to
the run that produced it. Everything in it is measured at the end of the run:
row counts from the database, relation sizes from Postgres, wall clock and peak
resident memory from the process. Nothing is estimated and nothing is carried
over from a previous run.

The ``quotable`` block is not decoration. This artifact records what was built,
not how well it retrieves, and the difference matters enough to state inside the
file rather than in a document beside it. There is no gold set: no recall, MRR
or nDCG figure has been computed for this project, and none may be inferred from
a corpus having been indexed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from django.db import connection

from signaldesk.analytics.signals import history_root
from signaldesk.core.logging import get_logger
from signaldesk.core.provenance import code_sha, peak_rss_bytes
from signaldesk.web.documents.models import (
    EMBEDDING_DIMENSIONS,
    ChunkEmbedding,
    LabelChunk,
    LabelChunkOccurrence,
    LabelDocument,
    LabelSection,
)

log = get_logger(__name__)

#: What the artifact says about retrieval quality, which is nothing.
WITHHELD_REASON = (
    "No gold set exists. evals/golden/ holds no judgements, so no recall, MRR or "
    "nDCG figure has been computed for this project and none may be inferred "
    "from a corpus having been indexed. This artifact records what was built."
)


def _relation_bytes(relation: str) -> int:
    """Total on-disk size of one table or index, or zero if it does not exist.

    Read from Postgres rather than estimated from row counts, because the point
    of recording it is to know what the index actually costs.
    """
    with connection.cursor() as cursor:
        cursor.execute("SELECT to_regclass(%s)", [relation])
        row = cursor.fetchone()
        if row is None or row[0] is None:
            return 0
        cursor.execute("SELECT pg_total_relation_size(%s)", [relation])
        size = cursor.fetchone()
        return int(size[0]) if size else 0


def _index_bytes(index: str) -> int:
    with connection.cursor() as cursor:
        cursor.execute("SELECT to_regclass(%s)", [index])
        row = cursor.fetchone()
        if row is None or row[0] is None:
            return 0
        cursor.execute("SELECT pg_relation_size(%s)", [index])
        size = cursor.fetchone()
        return int(size[0]) if size else 0


def _directory_bytes(path: Path) -> int:
    """Size of the sparse index directory, or zero if it is not there."""
    if not path.is_dir():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


@dataclass(frozen=True, slots=True)
class IndexRun:
    """One build, and everything needed to say what it produced."""

    run_id: str
    created_at: str
    params: dict[str, Any]
    corpus: dict[str, Any]
    dense: dict[str, Any]
    sparse: dict[str, Any]
    storage: dict[str, Any]
    seconds: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact": "index",
            "run_id": self.run_id,
            "created_at": self.created_at,
            "code_sha": code_sha(),
            "params": self.params,
            "corpus": self.corpus,
            "dense": self.dense,
            "sparse": self.sparse,
            "storage": self.storage,
            "performance": {
                "seconds": round(self.seconds, 2),
                "peak_rss_bytes": peak_rss_bytes(),
            },
            "quotable": {
                "measured": [
                    "corpus counts",
                    "chunk and embedding counts",
                    "index sizes",
                    "wall clock and peak memory",
                ],
                "withheld": ["retrieval accuracy"],
                "withheld_reason": WITHHELD_REASON,
            },
        }


def new_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def collect(
    *,
    run_id: str,
    params: dict[str, Any],
    chunking: dict[str, Any],
    embedding: dict[str, Any],
    sparse_chunks: int,
    sparse_path: Path,
    seconds: float,
) -> IndexRun:
    """Measure the state of the index and assemble the record.

    Counts are read back from the database at the end rather than accumulated
    during the run. An accumulated total describes what the code believed it did;
    a count describes what is there, and when the two disagree the second is the
    one worth having.
    """
    model = str(embedding.get("model", ""))
    return IndexRun(
        run_id=run_id,
        created_at=datetime.now(UTC).isoformat(),
        params=params,
        corpus={
            "label_documents": LabelDocument.objects.count(),
            "label_sections": LabelSection.objects.count(),
            "chunks_total": LabelChunk.objects.count(),
            "chunk_occurrences": LabelChunkOccurrence.objects.count(),
            **chunking,
        },
        dense={
            **embedding,
            "embeddings_total": ChunkEmbedding.objects.filter(model=model).count(),
            "embedding_dimensions": EMBEDDING_DIMENSIONS,
        },
        sparse={
            "chunks_indexed": sparse_chunks,
            "index_bytes": _directory_bytes(sparse_path),
            "path": str(sparse_path),
        },
        storage={
            "label_chunk_bytes": _relation_bytes("label_chunk"),
            "chunk_embedding_bytes": _relation_bytes("chunk_embedding"),
            "hnsw_index_bytes": _index_bytes("chunk_embedding_hnsw"),
        },
        seconds=seconds,
    )


def write(run: IndexRun, *, root: Path | None = None) -> Path:
    """Write the record under ``evals/history/`` and return its path.

    One file per run, named for the run. Nothing overwrites a previous build:
    two indexes built from different corpora are two facts, and collapsing them
    into one file would make the older one unquotable without saying so.
    """
    target = (root or history_root()) / f"index_{run.run_id}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(run.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log.info("rag.index.artifact", path=str(target), run_id=run.run_id)
    return target
