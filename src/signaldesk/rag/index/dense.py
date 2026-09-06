"""The dense half of retrieval, over pgvector.

Vectors live in Postgres next to the corpus they came from rather than in a
separate service. That is a decision the stack already made and this module
inherits it; the practical consequence is that a filtered search - "only this
drug's labels" - is a join rather than a second system to keep consistent.

Two Postgres settings are applied per query and neither is optional:

``hnsw.ef_search``
    How wide the graph walk is. Below the number of rows wanted it cannot return
    them. Read from ``hnsw_ef_search`` in the settings object.

``hnsw.iterative_scan``
    Without it a filtered HNSW search post-filters: the graph returns its best
    ``ef_search`` rows overall, the WHERE clause removes most of them, and the
    query returns far fewer than asked for while looking like it simply found
    less. pgvector 0.8 added iterative scan to fix exactly this, the installed
    server is 0.8.6, and every filtered search here sets it.

Both are set with SET LOCAL inside the caller's transaction, so they last for
that statement and change nothing for anyone else.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from django.db import connection, transaction
from pgvector.django import CosineDistance

from signaldesk.core.config import Settings, get_settings
from signaldesk.core.logging import get_logger
from signaldesk.rag.embed import assert_dimensions, normalize_rows
from signaldesk.stats.types import FloatArray
from signaldesk.web.documents.models import EMBEDDING_DIMENSIONS, ChunkEmbedding

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class DenseHit:
    """One vector match, with the distance it was chosen on."""

    chunk_id: int
    #: Cosine distance in [0, 2]. Lower is nearer. Kept rather than converted to
    #: a similarity so that what is stored is what the index actually ranked on.
    distance: float


def store(
    chunk_ids: Sequence[int],
    vectors: FloatArray,
    *,
    model: str,
    model_revision: str = "",
) -> int:
    """Write one model's vectors for ``chunk_ids``. Returns rows written.

    Idempotent on ``(chunk, model)``: re-running replaces rather than
    accumulating, so a resumed embed run cannot leave a chunk with two vectors
    from the same model and no way to tell which was current.
    """
    if len(chunk_ids) != vectors.shape[0]:
        message = (
            f"{len(chunk_ids)} chunk ids against {vectors.shape[0]} vectors; "
            "they correspond by position"
        )
        raise ValueError(message)
    if not len(chunk_ids):
        return 0

    assert_dimensions(vectors, EMBEDDING_DIMENSIONS)
    unit = normalize_rows(vectors)
    rows = [
        ChunkEmbedding(
            chunk_id=chunk_id,
            model=model,
            model_revision=model_revision,
            dimensions=EMBEDDING_DIMENSIONS,
            vector=unit[position],
        )
        for position, chunk_id in enumerate(chunk_ids)
    ]
    with transaction.atomic():
        written = ChunkEmbedding.objects.bulk_create(
            rows,
            update_conflicts=True,
            update_fields=["vector", "model_revision", "dimensions"],
            unique_fields=["chunk", "model"],
        )
    log.info("rag.dense.stored", rows=len(written), model=model)
    return len(written)


def _apply_scan_settings(settings: Settings) -> None:
    """Set ef_search and iterative scan for the current transaction."""
    with connection.cursor() as cursor:
        cursor.execute(f"SET LOCAL hnsw.ef_search = {int(settings.hnsw_ef_search)}")
        # relaxed_order rather than strict_order: strict ordering costs more and
        # buys exact distance ordering, which fusion discards anyway - RRF reads
        # ranks, not distances.
        cursor.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")


def search(
    query_vector: FloatArray,
    k: int,
    *,
    model: str,
    document_ids: Sequence[int] | None = None,
    settings: Settings | None = None,
) -> tuple[DenseHit, ...]:
    """The ``k`` nearest chunks to ``query_vector``, nearest first.

    ``document_ids`` restricts the search to chunks occurring in those label
    documents, which is how a per-drug question is asked: resolve the drug to
    its labels, then search only those. Passing None searches the whole corpus.
    """
    settings = settings or get_settings()
    if k < 1:
        return ()

    unit = normalize_rows(query_vector.reshape(1, -1))
    assert_dimensions(unit, EMBEDDING_DIMENSIONS)

    queryset = ChunkEmbedding.objects.filter(model=model)
    if document_ids is not None:
        if not document_ids:
            # An empty restriction is not the same question as an unrestricted
            # one. A drug whose labels are not in the corpus has no chunks to
            # search, and returning the corpus-wide best matches would answer a
            # question nobody asked.
            return ()
        queryset = queryset.filter(chunk__occurrences__section__document_id__in=document_ids)

    with transaction.atomic():
        _apply_scan_settings(settings)
        rows = list(
            queryset.annotate(distance=CosineDistance("vector", unit[0]))
            .order_by("distance")
            .values_list("chunk_id", "distance")
            .distinct()[:k]
        )
    return tuple(
        DenseHit(chunk_id=int(chunk_id), distance=float(distance)) for chunk_id, distance in rows
    )
