"""Walking the label corpus into chunks, and the sparse index over them.

Neither step needs a model, so both run anywhere the database does - including
continuous integration, which has no torch. The dense step is the one that needs
an encoder and it is not here.

Resumability is structural rather than a manifest table. A section that already
has occurrence rows has been chunked, so a re-run skips it; there is nothing to
record separately and therefore nothing that can disagree with the data. The
chunking itself is deterministic, so a skipped section and a re-chunked one
would produce identical rows anyway - the skip is for the time the sentence
splitter costs over the corpus, not for correctness.

Deduplication happens twice and the two are not the same. Section texts repeat
verbatim across manufacturers, so identical texts are chunked once and the
result reused. Then chunks themselves repeat, because two labels that differ in
one paragraph share every other window, so chunks are stored once per distinct
(section code, text) and pointed at from every section they occur in.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from django.db import transaction

from signaldesk.core.config import Settings, get_settings
from signaldesk.core.logging import get_logger
from signaldesk.rag import chunking
from signaldesk.rag.index import sparse
from signaldesk.web.documents.models import LabelChunk, LabelChunkOccurrence, LabelSection

log = get_logger(__name__)

#: Sections pulled into memory at once. Section lengths vary by orders of
#: magnitude and the longest are very long, so this is kept small deliberately.
SECTION_BATCH = 200


@dataclass(frozen=True, slots=True)
class ChunkRun:
    """What one chunking pass did."""

    sections_seen: int
    sections_chunked: int
    sections_skipped: int
    distinct_texts: int
    chunks_created: int
    occurrences_created: int
    seconds: float

    def as_dict(self) -> dict[str, object]:
        return {
            "sections_seen": self.sections_seen,
            "sections_chunked": self.sections_chunked,
            "sections_skipped_already_chunked": self.sections_skipped,
            "distinct_section_texts": self.distinct_texts,
            "chunks_created": self.chunks_created,
            "occurrences_created": self.occurrences_created,
            "seconds": round(self.seconds, 2),
        }


def _chunk_ids_for(shas: list[str]) -> dict[str, int]:
    return dict(LabelChunk.objects.filter(sha256__in=shas).values_list("sha256", "id"))


def chunk_corpus(settings: Settings | None = None, *, force: bool = False) -> ChunkRun:
    """Chunk every label section that has not been chunked yet.

    ``force`` re-chunks sections that already have occurrences, for use after a
    change to the chunking parameters or the splitter. Without it the skip makes
    such a change invisible, which is the failure mode a resumable step usually
    has.
    """
    settings = settings or get_settings()
    started = time.monotonic()

    already: set[int] = set()
    if not force:
        already = set(LabelChunkOccurrence.objects.values_list("section_id", flat=True).distinct())

    # Section text hash to the chunk hashes it produced. Identical texts are
    # chunked once; this holds hashes rather than text, so it stays small.
    chunked_texts: dict[str, list[tuple[str, int, str, int]]] = {}
    seen = chunked = skipped = created_chunks = created_occurrences = 0

    queryset = LabelSection.objects.order_by("id").values_list("id", "section_code", "text")
    for section_id, section_code, text in queryset.iterator(chunk_size=SECTION_BATCH):
        seen += 1
        if section_id in already:
            skipped += 1
            continue

        key = chunking.chunk_sha256(section_code, chunking.normalize(text))
        cached = chunked_texts.get(key)
        if cached is None:
            chunks = chunking.chunk_section(
                text,
                section_code,
                target_tokens=settings.chunk_target_tokens,
                overlap_tokens=settings.chunk_overlap_tokens,
            )
            cached = [
                (chunk.sha256, chunk.ordinal, chunk.text, chunk.token_estimate) for chunk in chunks
            ]
            chunked_texts[key] = cached

        if not cached:
            chunked += 1
            continue

        with transaction.atomic():
            rows = [
                LabelChunk(sha256=sha, section_code=section_code, text=body, token_estimate=tokens)
                for sha, _ordinal, body, tokens in cached
            ]
            # ignore_conflicts rather than update: a chunk is identified by a
            # hash of its own content, so a row that already exists is already
            # correct and rewriting it would be work with no possible effect.
            new_chunks = LabelChunk.objects.bulk_create(rows, ignore_conflicts=True)
            created_chunks += sum(1 for chunk in new_chunks if chunk.pk is not None)

            ids = _chunk_ids_for([sha for sha, _, _, _ in cached])
            occurrences = [
                LabelChunkOccurrence(chunk_id=ids[sha], section_id=section_id, ordinal=ordinal)
                for sha, ordinal, _, _ in cached
            ]
            new_occurrences = LabelChunkOccurrence.objects.bulk_create(
                occurrences, ignore_conflicts=True
            )
            created_occurrences += len(new_occurrences)
        chunked += 1

        if chunked % 500 == 0:
            log.info("rag.chunk.progress", sections=chunked, of=seen, chunks=created_chunks)

    run = ChunkRun(
        sections_seen=seen,
        sections_chunked=chunked,
        sections_skipped=skipped,
        distinct_texts=len(chunked_texts),
        chunks_created=created_chunks,
        occurrences_created=created_occurrences,
        seconds=time.monotonic() - started,
    )
    log.info("rag.chunk.done", **run.as_dict())
    return run


def build_sparse_index(settings: Settings | None = None) -> tuple[int, str]:
    """Build the BM25 index over every stored chunk. Returns count and path.

    Rebuilt whole rather than updated. bm25s computes corpus statistics at index
    time, so an incrementally extended index would score new documents against
    the term frequencies of an older corpus.
    """
    settings = settings or get_settings()
    rows = list(LabelChunk.objects.order_by("id").values_list("id", "text"))
    if not rows:
        message = (
            "there are no chunks to index. Run 'signaldesk index chunk' first; "
            "this is not a corpus with no lexical content."
        )
        raise sparse.SparseIndexError(message)

    path = sparse.build(
        [chunk_id for chunk_id, _ in rows],
        [text for _, text in rows],
        settings=settings,
    )
    return len(rows), str(path)
