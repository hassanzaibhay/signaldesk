"""The BM25 index: building it, loading it, and what it says when it is absent.

bm25s is a base dependency rather than part of the ``ml`` extra, so this runs in
continuous integration where nothing torch-shaped does. It writes to a temporary
directory and never touches ``DATA_DIR``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from signaldesk.rag.index.sparse import (
    IDS_FILENAME,
    SparseIndex,
    SparseIndexError,
    build,
    indexed_count,
)

pytestmark = pytest.mark.unit

CHUNK_IDS = [11, 22, 33, 44]
TEXTS = [
    "Lactic acidosis has been reported in patients receiving metformin hydrochloride.",
    "Hepatic failure and jaundice have been observed with acetaminophen overdose.",
    "Serious rash including Stevens Johnson syndrome has occurred.",
    "Patients should be advised to report any unusual bleeding.",
]


@pytest.fixture
def index(tmp_path: Path) -> SparseIndex:
    build(CHUNK_IDS, TEXTS, path=tmp_path / "bm25")
    return SparseIndex.load(path=tmp_path / "bm25")


class TestBuilding:
    def test_the_index_holds_every_chunk(self, index: SparseIndex) -> None:
        assert len(index) == len(CHUNK_IDS)

    def test_ids_are_written_beside_the_index(self, tmp_path: Path) -> None:
        """A row position means nothing without the mapping that produced it."""
        path = build(CHUNK_IDS, TEXTS, path=tmp_path / "bm25")

        assert (path / IDS_FILENAME).is_file()

    def test_mismatched_ids_and_texts_are_refused(self, tmp_path: Path) -> None:
        """Truncating to the shorter would index a corpus missing its tail."""
        with pytest.raises(SparseIndexError, match="same length"):
            build([1, 2], TEXTS, path=tmp_path / "bm25")

    def test_building_over_nothing_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(SparseIndexError, match="zero chunks"):
            build([], [], path=tmp_path / "bm25")


class TestCountingWithoutLoading:
    """What a run that did not build the index can still say about it."""

    def test_it_counts_the_chunks_the_index_covers(self, tmp_path: Path) -> None:
        build(CHUNK_IDS, TEXTS, path=tmp_path / "bm25")

        assert indexed_count(tmp_path / "bm25") == len(CHUNK_IDS)

    def test_an_absent_index_counts_as_nothing_rather_than_as_zero(self, tmp_path: Path) -> None:
        """Zero would read as an index over an empty corpus, which is not this."""
        assert indexed_count(tmp_path / "nothing-here") is None


class TestSearching:
    def test_a_lexical_match_is_found(self, index: SparseIndex) -> None:
        """The case dense retrieval is worst at: an exact drug name."""
        hits = index.search("metformin lactic acidosis", k=1)

        assert hits[0].chunk_id == 11

    def test_results_come_back_as_chunk_ids_not_row_positions(self, index: SparseIndex) -> None:
        hits = index.search("acetaminophen hepatic failure", k=2)

        assert hits[0].chunk_id == 22
        assert all(hit.chunk_id in CHUNK_IDS for hit in hits)

    def test_results_are_ordered_best_first(self, index: SparseIndex) -> None:
        hits = index.search("rash Stevens Johnson", k=4)

        assert hits[0].chunk_id == 33
        assert [hit.score for hit in hits] == sorted((hit.score for hit in hits), reverse=True)

    def test_asking_for_more_than_the_index_holds_returns_what_it_has(
        self, index: SparseIndex
    ) -> None:
        """Fewer results is a fact about the corpus, not a failure."""
        assert len(index.search("metformin", k=99)) == len(CHUNK_IDS)

    def test_asking_for_none_returns_none(self, index: SparseIndex) -> None:
        assert index.search("metformin", k=0) == ()

    def test_a_query_matching_nothing_still_returns_a_list(self, index: SparseIndex) -> None:
        """BM25 scores everything; nothing relevant is a ranking, not an error."""
        assert isinstance(index.search("zzzqqq nonexistent", k=2), tuple)


class TestLoading:
    def test_an_absent_index_is_distinguished_from_an_empty_one(self, tmp_path: Path) -> None:
        """ "Retrieval returned nothing" has to mean one thing."""
        with pytest.raises(SparseIndexError, match="no sparse index at"):
            SparseIndex.load(path=tmp_path / "never-built")

    def test_an_index_without_its_ids_is_refused(self, tmp_path: Path) -> None:
        """Its results would name the wrong rows, which is worse than no results."""
        path = build(CHUNK_IDS, TEXTS, path=tmp_path / "bm25")
        (path / IDS_FILENAME).unlink()

        with pytest.raises(SparseIndexError, match="cannot be resolved to chunks"):
            SparseIndex.load(path=path)
