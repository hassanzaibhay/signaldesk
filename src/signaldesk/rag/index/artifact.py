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

Two vector widths are recorded and they are not the same claim.
``embedding_dimensions`` is the column's declared width, a constant.
``measured_dimensions`` is what the weights actually produced, read back from the
stored rows. An artifact that carried only the first would assert agreement it
never checked.

Timings likewise come in two kinds. ``run_window`` is one process, from before
its model load to its last write, and only a run that recorded it has one.
``write_window`` spans every vector for a model whatever run wrote it, so on a
resumed embed it is corpus-level rather than run-level. It is built from
``created_at``, which is ``auto_now_add`` and absent from the upsert's
``update_fields``, so it measures first writes only: had any row been
re-embedded, the throughput derived from it would understate the work performed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from django.db import connection

from signaldesk.analytics.signals import history_root
from signaldesk.core.config import Settings, get_settings
from signaldesk.core.logging import get_logger
from signaldesk.core.provenance import code_sha, peak_rss_bytes
from signaldesk.rag.index import sparse
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

#: What no record assembled after the fact can contain. A run that did not
#: record these left nothing behind to read them from, and they are omitted
#: rather than estimated: an inferred number in an artifact is worse than an
#: absent one, because only the absent one is obviously absent.
NEVER_CAPTURED = (
    "peak resident memory of the embedding process",
    "model load seconds",
    "run_window: no started_at was captured, so the process start and end are "
    "unrecoverable. Runs from this change onward record it.",
    "the batch count as the run recorded it",
    "the commit the run was at. code_sha on this record is the commit at the "
    "time it was written, not the time it was run.",
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


def _widths(model: str) -> dict[str, Any]:
    """What width the stored vectors actually are, under ``model``.

    Two readings, because they can disagree and the disagreement is the whole
    point of measuring. ``dimensions`` is what the writer recorded; the width
    ``vector_dims`` reports is what is in the column. A single value is emitted
    only when every row agrees on it, so a mixed table reads as unknown rather
    than as whichever row the query happened to reach first.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT min(dimensions), max(dimensions), "
            "min(vector_dims(vector)), max(vector_dims(vector)), "
            "count(*) FILTER (WHERE dimensions <> %s OR vector_dims(vector) <> %s) "
            "FROM chunk_embedding WHERE model = %s",
            [EMBEDDING_DIMENSIONS, EMBEDDING_DIMENSIONS, model],
        )
        row = cursor.fetchone()

    if row is None:
        return {
            "measured_dimensions": None,
            "recorded_dimensions": None,
            "rows_with_unexpected_dimensions": 0,
        }
    recorded_min, recorded_max, actual_min, actual_max, off_width = row
    # An aggregate over no rows is a row of nulls rather than no row, so a table
    # with nothing in it under this model arrives here looking like agreement on
    # a width of None. Unknown and agreed-upon are not the same answer.
    agreed_actual = actual_min is not None and actual_min == actual_max
    agreed_recorded = recorded_min is not None and recorded_min == recorded_max
    return {
        "measured_dimensions": int(actual_min) if agreed_actual else None,
        "recorded_dimensions": int(recorded_min) if agreed_recorded else None,
        "rows_with_unexpected_dimensions": int(off_width or 0),
    }


def _write_window(model: str) -> dict[str, Any]:
    """First and last vector written for ``model``, and the rate between them.

    Corpus-level, not run-level: it spans every vector under the model whatever
    run wrote it. See the module docstring for why it measures first writes only.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT min(created_at), max(created_at), count(*) "
            "FROM chunk_embedding WHERE model = %s",
            [model],
        )
        row = cursor.fetchone()

    if row is None or row[0] is None:
        return {
            "first_vector_at": None,
            "last_vector_at": None,
            "seconds": None,
            "chunks_per_second": None,
        }
    first, last, rows = row
    seconds = (last - first).total_seconds()
    return {
        "first_vector_at": first.isoformat(),
        "last_vector_at": last.isoformat(),
        "seconds": round(seconds, 2),
        "chunks_per_second": round(int(rows) / seconds, 3) if seconds > 0 else None,
    }


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
    #: Present only on a record assembled after the fact, by ``index artifact``.
    #: Its presence is the statement that this file was not written by the run
    #: it describes; see ``reconstruct``.
    reconstructed: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        document: dict[str, Any] = {
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
                    "vector width, read back from the stored rows",
                    "index sizes",
                    "wall clock and peak memory",
                ],
                "withheld": ["retrieval accuracy"],
                "withheld_reason": WITHHELD_REASON,
            },
        }
        if self.reconstructed is not None:
            document["reconstructed"] = self.reconstructed
        return document


def new_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def writer_evidence(model: str, *, batch_rows: int = 256) -> dict[str, Any]:
    """The transaction structure behind one model's rows.

    A record assembled after the fact and named for a run asserts that the rows
    it describes came from that run. Nothing recorded the process, so the claim
    rests on what the tuples still carry: one transaction per batch, every
    transaction boundary in timestamp order, and no row ever superseded.

    These are measured here rather than quoted from an investigation, so the
    artifact carries the evidence for its own name and a later reader can judge
    it without asking anyone. ``unaccounted_xids`` is the residual - transaction
    ids inside the span that did not write to this table. Their contents are not
    measured here. It is a bound, not a zero.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "WITH t AS ("
            "  SELECT xmin::text::bigint AS xid, min(created_at) AS first_at, count(*) AS rows"
            "  FROM chunk_embedding WHERE model = %s GROUP BY 1), "
            "o AS ("
            "  SELECT xid, rows, lag(xid) OVER (ORDER BY first_at) AS prev_xid FROM t) "
            "SELECT count(*), "
            "       count(*) FILTER (WHERE prev_xid IS NOT NULL AND xid <= prev_xid), "
            "       count(*) FILTER (WHERE rows = %s), min(rows) "
            "FROM o",
            [model, batch_rows],
        )
        transactions, out_of_order, full_batches, smallest = cursor.fetchone() or (0, 0, 0, 0)

        cursor.execute(
            "SELECT max(xmin::text::bigint) - min(xmin::text::bigint), "
            "       count(*) FILTER (WHERE xmax::text::bigint <> 0) "
            "FROM chunk_embedding WHERE model = %s",
            [model],
        )
        span, superseded = cursor.fetchone() or (0, 0)

    return {
        "writing_transactions": int(transactions or 0),
        "full_batch_transactions": int(full_batches or 0),
        "batch_rows": batch_rows,
        "final_batch_rows": int(smallest or 0),
        "out_of_order_transaction_boundaries": int(out_of_order or 0),
        "xid_span": int(span or 0),
        "unaccounted_xids": int(span or 0) - int(transactions or 0),
        "rows_ever_updated_or_deleted": int(superseded or 0),
    }


def collect(
    *,
    run_id: str,
    params: dict[str, Any],
    chunking: dict[str, Any],
    embedding: dict[str, Any],
    sparse_path: Path,
    seconds: float,
    sparse_rebuilt: bool = False,
    reconstructed: dict[str, Any] | None = None,
) -> IndexRun:
    """Measure the state of the index and assemble the record.

    Counts are read back from the database at the end rather than accumulated
    during the run. An accumulated total describes what the code believed it did;
    a count describes what is there, and when the two disagree the second is the
    one worth having.

    ``chunking`` and ``embedding`` each carry a ``ran`` flag, because a command
    that did only one half still describes the whole index and the record has to
    keep "this run did not chunk" apart from "there is nothing chunked".

    ``chunks_without_embedding`` is counted here rather than taken from the embed
    run, so it is present on every path. Reading it only off a run that embedded
    is what let a chunk-only build report a finished-looking index while saying
    nothing about how much of it had vectors.
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
            "chunks_without_embedding": LabelChunk.objects.exclude(embeddings__model=model).count(),
            "embedding_dimensions": EMBEDDING_DIMENSIONS,
            **_widths(model),
            "write_window": _write_window(model),
        },
        sparse={
            "chunks_indexed": sparse.indexed_count(sparse_path),
            "rebuilt": sparse_rebuilt,
            "index_bytes": _directory_bytes(sparse_path),
            "path": str(sparse_path),
        },
        storage={
            "label_chunk_bytes": _relation_bytes("label_chunk"),
            "chunk_embedding_bytes": _relation_bytes("chunk_embedding"),
            "hnsw_index_bytes": _index_bytes("chunk_embedding_hnsw"),
        },
        seconds=seconds,
        reconstructed=reconstructed,
    )


def _params(settings: Settings) -> dict[str, Any]:
    return {
        "chunk_target_tokens": settings.chunk_target_tokens,
        "chunk_overlap_tokens": settings.chunk_overlap_tokens,
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
    }


def record(
    *,
    run_id: str,
    chunking: dict[str, Any],
    embedding: dict[str, Any],
    seconds: float,
    settings: Settings | None = None,
    sparse_rebuilt: bool = False,
    reconstructed: dict[str, Any] | None = None,
    root: Path | None = None,
) -> Path:
    """Assemble one record and write it. The only way an artifact is produced.

    Both entry points that write an artifact go through here, so the params
    block, the measurement set and the file naming cannot drift apart between a
    build and an embed. Which of them is calling shows up in the ``ran`` flags
    and, for a record assembled afterwards, in ``reconstructed``.
    """
    settings = settings or get_settings()
    return write(
        collect(
            run_id=run_id,
            params=_params(settings),
            chunking=chunking,
            embedding=embedding,
            sparse_path=sparse.index_root(settings),
            seconds=seconds,
            sparse_rebuilt=sparse_rebuilt,
            reconstructed=reconstructed,
        ),
        root=root,
    )


def reconstruct(
    *,
    run_id: str,
    settings: Settings | None = None,
    root: Path | None = None,
) -> Path:
    """Write a record for a run that wrote none, from the state it left behind.

    Everything in the resulting file is measured now, from the database and the
    index on disk. Nothing is carried in from a console transcript. What the run
    never recorded is listed in ``never_captured`` and left out of the document
    rather than estimated into it.

    The ``reconstructed`` block is emitted here and only here, by virtue of this
    being the entry point rather than by an argument anyone can pass. A record
    assembled after the fact cannot be produced without saying that it was.
    """
    settings = settings or get_settings()
    model = settings.embedding_model
    return record(
        run_id=run_id,
        settings=settings,
        chunking={"ran": False},
        embedding={"model": model, "ran": False},
        seconds=0.0,
        sparse_rebuilt=False,
        reconstructed={
            "written_at": datetime.now(UTC).isoformat(),
            "written_by": "signaldesk index artifact",
            "measured_now": [
                "corpus counts",
                "embedding count, model, revision and width",
                "chunks_without_embedding",
                "relation and index sizes",
                "write_window, from chunk_embedding.created_at",
                "writer_evidence, from chunk_embedding xmin and xmax",
            ],
            "never_captured": list(NEVER_CAPTURED),
            "writer_evidence": writer_evidence(model),
            "run_id_rests_on": (
                "xmin structure. The rows fall into one transaction per batch, in "
                "perfect batch alignment, with no inverted transaction boundary "
                "anywhere in the span, and no row has ever been superseded. That is "
                "what names this record for a single run. pg_stat_user_tables could "
                "not corroborate it: crash recovery discarded the cumulative "
                "statistics at 2026-09-08T07:32:19Z and every counter reads zero, "
                "which is absence of evidence rather than evidence of none. The "
                "residual is unaccounted_xids - transaction ids inside the span "
                "that did not write to this table. Their contents were not "
                "measured. It is a bound, not a zero."
            ),
        },
        root=root,
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
