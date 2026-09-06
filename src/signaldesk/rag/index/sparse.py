"""The lexical half of retrieval, over bm25s.

BM25 is here because dense retrieval is bad at exactly what this corpus is full
of: exact drug names, exact reaction terms, and numbers. An embedding of
"thrombotic thrombocytopenic purpura" sits near a lot of haematology; a lexical
match sits on the phrase. The two fail differently, which is the entire argument
for fusing them rather than picking one.

bm25s writes a small directory rather than a database table, so the index lives
under ``DATA_DIR`` beside the other build output. Chunk identifiers are stored
next to it as their own file: bm25s indexes by row position, and a position is
only meaningful with the mapping that produced it. Keeping that mapping inside
this module means no caller has to know that bm25s ever saw a row number.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from signaldesk.core.config import Settings, get_settings
from signaldesk.core.errors import SignalDeskError
from signaldesk.core.logging import get_logger

log = get_logger(__name__)

#: Written beside the bm25s files. Row position to chunk id, in index order.
IDS_FILENAME = "chunk_ids.json"


class SparseIndexError(SignalDeskError):
    """The sparse index is absent, unreadable, or inconsistent with its ids."""


@dataclass(frozen=True, slots=True)
class SparseHit:
    """One lexical match."""

    chunk_id: int
    score: float


def index_root(settings: Settings | None = None) -> Path:
    """Where the sparse index is written."""
    settings = settings or get_settings()
    return settings.data_dir / "index" / "bm25"


def _tokenizer() -> Any:
    """The English stemmer bm25s tokenises through.

    Imported here rather than at module scope so that a machine without
    PyStemmer still imports this module and fails at use with a clear error.
    """
    import Stemmer  # type: ignore[import-not-found]

    return Stemmer.Stemmer("english")


def build(
    chunk_ids: list[int],
    texts: list[str],
    *,
    settings: Settings | None = None,
    path: Path | None = None,
) -> Path:
    """Build and persist a BM25 index over ``texts``.

    ``chunk_ids`` and ``texts`` correspond by position and that correspondence
    is the index. A mismatch in length is refused rather than truncated to the
    shorter of the two: a silently shortened index returns real-looking results
    for a corpus that is missing its tail.
    """
    if len(chunk_ids) != len(texts):
        message = (
            f"{len(chunk_ids)} chunk ids against {len(texts)} texts; they index "
            "each other by position and must be the same length"
        )
        raise SparseIndexError(message)
    if not chunk_ids:
        message = "refusing to build a sparse index over zero chunks"
        raise SparseIndexError(message)

    import bm25s  # type: ignore[import-untyped]

    target = path or index_root(settings)
    target.mkdir(parents=True, exist_ok=True)

    tokens = bm25s.tokenize(texts, stopwords="en", stemmer=_tokenizer(), show_progress=False)
    retriever = bm25s.BM25()
    retriever.index(tokens, show_progress=False)
    retriever.save(str(target))
    (target / IDS_FILENAME).write_text(json.dumps(chunk_ids), encoding="utf-8")

    log.info("rag.sparse.built", chunks=len(chunk_ids), path=str(target))
    return target


class SparseIndex:
    """A loaded BM25 index, ready to answer queries."""

    def __init__(self, retriever: Any, chunk_ids: list[int]) -> None:
        self._retriever = retriever
        self._chunk_ids = chunk_ids

    def __len__(self) -> int:
        return len(self._chunk_ids)

    @classmethod
    def load(cls, *, settings: Settings | None = None, path: Path | None = None) -> SparseIndex:
        """Load the index, or say precisely what is missing.

        An absent index and an index over an empty corpus are different states.
        This distinguishes them, because "retrieval returned nothing" is a
        sentence that has to mean one thing.
        """
        import bm25s

        target = path or index_root(settings)
        ids_path = target / IDS_FILENAME
        if not target.is_dir():
            message = (
                f"no sparse index at {target}. It is built by 'signaldesk index build'; "
                "this is not a corpus with no lexical matches."
            )
            raise SparseIndexError(message)
        if not ids_path.is_file():
            message = (
                f"the sparse index at {target} has no {IDS_FILENAME}. Its row "
                "positions cannot be resolved to chunks, so any result it "
                "returned would name the wrong rows. Rebuild it."
            )
            raise SparseIndexError(message)

        chunk_ids: list[int] = json.loads(ids_path.read_text(encoding="utf-8"))
        retriever = bm25s.BM25.load(str(target))
        return cls(retriever, chunk_ids)

    def search(self, query: str, k: int) -> tuple[SparseHit, ...]:
        """The ``k`` best lexical matches for ``query``, best first.

        Returns fewer than ``k`` when the index is smaller, which is a fact
        about the corpus rather than a failure. bm25s is asked for no more than
        it holds because asking for more raises.
        """
        import bm25s

        if k < 1 or not self._chunk_ids:
            return ()
        tokens = bm25s.tokenize(query, stopwords="en", stemmer=_tokenizer(), show_progress=False)
        wanted = min(k, len(self._chunk_ids))
        positions, scores = self._retriever.retrieve(tokens, k=wanted, show_progress=False)
        return tuple(
            SparseHit(chunk_id=self._chunk_ids[int(position)], score=float(score))
            for position, score in zip(positions[0], scores[0], strict=True)
        )
