"""The whole pipeline, end to end, over a synthetic fixture.

Real Postgres, real pgvector with a real HNSW index, real SQL, a real bm25s
index, real fusion and real metrics. The only stand-ins are the two encoders,
because neither continuous integration job installs torch and a test that needed
it would not run where it matters.

That is the point of the split. Everything that decides anything is exercised
here; what is left for the model adapters is turning a list of strings into an
array. This suite is what "the harness ships ready to run" means: it runs, over
judgements that are synthetic and are labelled as such, and it computes the same
metrics a real gold set would go through.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from django.db.models import Count
from tests.conftest import HashingDocumentEncoder, HashingEncoder, OverlapCrossEncoder

from signaldesk.core.config import get_settings
from signaldesk.evals.retrieval import suite
from signaldesk.evals.retrieval.gold import GoldSet, RetrievalGoldSetError, parse_judgements
from signaldesk.rag import chunking
from signaldesk.rag.embed import EmbeddingError, embed_texts
from signaldesk.rag.index import artifact, dense, sparse
from signaldesk.rag.index.corpus import (
    build_sparse_index,
    chunk_corpus,
    embed_pending,
    pending_chunks,
)
from signaldesk.rag.retrieve import retrieve
from signaldesk.web.documents.models import (
    EMBEDDING_DIMENSIONS,
    ChunkEmbedding,
    LabelChunk,
    LabelChunkOccurrence,
    LabelDocument,
    LabelSection,
)

pytestmark = [pytest.mark.integration, pytest.mark.django_db(transaction=True)]

MODEL = "stub/hashing-encoder"

#: Four short sections across two labels. Short enough that each is one chunk,
#: so a judgement can name a chunk hash that is predictable from the text.
SECTIONS = [
    (
        "boxed_warning",
        "Lactic acidosis is a rare but serious metabolic complication that can "
        "occur due to metformin accumulation during treatment.",
    ),
    (
        "adverse_reactions",
        "The most common adverse reactions were diarrhoea, nausea and flatulence.",
    ),
    (
        "warnings",
        "Hepatic failure resulting in death has been reported with acetaminophen "
        "at doses exceeding the maximum daily amount.",
    ),
    (
        "adverse_reactions",
        "Serious skin reactions including Stevens Johnson syndrome have occurred rarely.",
    ),
]


@pytest.fixture
def corpus(tmp_path):  # type: ignore[no-untyped-def]
    """Two labels, four sections, chunked, embedded and indexed."""
    document = LabelDocument.objects.create(
        set_id="1f2e3d4c-0000-4a11-9f00-00000000feed", brand_names=["FIXTURAMIN"]
    )
    for ordinal, (code, text) in enumerate(SECTIONS):
        LabelSection.objects.create(
            document=document, section_code=code, ordinal=ordinal, text=text
        )

    chunk_corpus()

    rows = list(LabelChunk.objects.order_by("id").values_list("id", "text"))
    vectors = embed_texts(
        HashingEncoder(), [text for _, text in rows], expected_dimensions=EMBEDDING_DIMENSIONS
    )
    dense.store([chunk_id for chunk_id, _ in rows], vectors, model=MODEL)

    index_path = tmp_path / "bm25"
    sparse.build([chunk_id for chunk_id, _ in rows], [text for _, text in rows], path=index_path)
    return sparse.SparseIndex.load(path=index_path)


def _sha(section_code: str, text: str) -> str:
    return chunking.chunk_sha256(section_code, chunking.normalize(text))


class TestChunkingTheCorpus:
    def test_every_section_is_chunked(self, corpus) -> None:  # type: ignore[no-untyped-def]
        assert LabelChunk.objects.count() == len(SECTIONS)

    def test_chunk_hashes_are_a_function_of_the_text_and_the_section(self, corpus) -> None:  # type: ignore[no-untyped-def]
        """What lets a gold set name a chunk that survives a rebuild."""
        expected = {_sha(code, text) for code, text in SECTIONS}

        assert set(LabelChunk.objects.values_list("sha256", flat=True)) == expected

    def test_a_second_pass_adds_nothing(self, corpus) -> None:  # type: ignore[no-untyped-def]
        """Resumable: an interrupted run continues rather than duplicating."""
        before = LabelChunk.objects.count()

        run = chunk_corpus()

        assert LabelChunk.objects.count() == before
        assert run.sections_chunked == 0
        assert run.sections_skipped == len(SECTIONS)

    def test_it_counts_the_chunks_it_created(self, corpus) -> None:  # type: ignore[no-untyped-def]
        """The counter is the rows in the table, not the rows offered to it.

        bulk_create with ignore_conflicts returns objects carrying no primary
        key, so a count taken from what it hands back is structurally zero
        however much work the run did.
        """
        run = chunk_corpus(force=True)

        assert LabelChunk.objects.count() == len(SECTIONS)
        assert run.chunks_created == 0
        assert run.occurrences_created == 0

    def test_a_first_pass_counts_every_row_it_put_there(self) -> None:
        """Nothing exists beforehand, so created must equal what is there after."""
        document = LabelDocument.objects.create(set_id="counted-0000-4a11-9f00-000000000003")
        for ordinal, (code, text) in enumerate(SECTIONS):
            LabelSection.objects.create(
                document=document, section_code=code, ordinal=ordinal, text=text
            )

        run = chunk_corpus()

        assert run.chunks_created == LabelChunk.objects.count() > 0
        assert run.occurrences_created == LabelChunkOccurrence.objects.count() > 0

    def test_identical_text_across_labels_is_stored_once(self) -> None:
        """Manufacturers relabel the same generic; the sections repeat verbatim."""
        shared = "Lactic acidosis has been reported."
        for set_id in ("aaa", "bbb"):
            document = LabelDocument.objects.create(set_id=set_id)
            LabelSection.objects.create(
                document=document, section_code="warnings", ordinal=0, text=shared
            )

        chunk_corpus()

        assert LabelChunk.objects.count() == 1
        assert LabelChunk.objects.first().occurrences.count() == 2


class TestDenseSearch:
    def test_the_nearest_chunk_is_the_one_about_the_query(self, corpus) -> None:  # type: ignore[no-untyped-def]
        query = embed_texts(
            HashingEncoder(),
            ["metformin lactic acidosis"],
            expected_dimensions=EMBEDDING_DIMENSIONS,
        )

        hits = dense.search(query[0], 4, model=MODEL)

        assert LabelChunk.objects.get(id=hits[0].chunk_id).section_code == "boxed_warning"

    def test_restricting_to_no_documents_returns_nothing(self, corpus) -> None:  # type: ignore[no-untyped-def]
        """A drug with no labels in the corpus has nothing to search.

        Returning the corpus-wide best matches would answer a different question
        from the one asked, and would do it invisibly.
        """
        query = embed_texts(
            HashingEncoder(), ["anything"], expected_dimensions=EMBEDDING_DIMENSIONS
        )

        assert dense.search(query[0], 4, model=MODEL, document_ids=[]) == ()

    def test_restricting_to_a_document_keeps_its_chunks(self, corpus) -> None:  # type: ignore[no-untyped-def]
        document_id = LabelDocument.objects.first().id
        query = embed_texts(
            HashingEncoder(), ["lactic acidosis"], expected_dimensions=EMBEDDING_DIMENSIONS
        )

        hits = dense.search(query[0], 10, model=MODEL, document_ids=[document_id])

        assert len(hits) == len(SECTIONS)

    def test_storing_is_idempotent_on_chunk_and_model(self, corpus) -> None:  # type: ignore[no-untyped-def]
        """A resumed embed run must not leave two vectors with no current one."""
        rows = list(LabelChunk.objects.order_by("id").values_list("id", "text"))
        vectors = embed_texts(
            HashingEncoder(), [text for _, text in rows], expected_dimensions=EMBEDDING_DIMENSIONS
        )

        dense.store([chunk_id for chunk_id, _ in rows], vectors, model=MODEL)

        from signaldesk.web.documents.models import ChunkEmbedding

        assert ChunkEmbedding.objects.filter(model=MODEL).count() == len(rows)


class TestTheAssembledPipeline:
    def test_retrieval_leads_with_the_relevant_chunk(self, corpus) -> None:  # type: ignore[no-untyped-def]
        results = retrieve(
            "metformin lactic acidosis",
            query_encoder=HashingEncoder(),
            cross_encoder=OverlapCrossEncoder(),
            sparse_index=corpus,
            embedding_model=MODEL,
        )

        assert results[0].section_code == "boxed_warning"
        assert "Lactic acidosis" in results[0].text

    def test_results_carry_where_each_retriever_found_them(self, corpus) -> None:  # type: ignore[no-untyped-def]
        results = retrieve(
            "hepatic failure acetaminophen",
            query_encoder=HashingEncoder(),
            cross_encoder=OverlapCrossEncoder(),
            sparse_index=corpus,
            embedding_model=MODEL,
        )

        assert results[0].dense_rank is not None
        assert results[0].sparse_rank is not None
        assert results[0].rerank_score is not None

    def test_the_cutoff_is_the_configured_rerank_depth(self, corpus) -> None:  # type: ignore[no-untyped-def]
        results = retrieve(
            "reactions",
            query_encoder=HashingEncoder(),
            cross_encoder=OverlapCrossEncoder(),
            sparse_index=corpus,
            embedding_model=MODEL,
        )

        assert len(results) <= get_settings().rerank_top_k

    def test_omitting_the_cross_encoder_is_a_configuration_not_a_failure(self, corpus) -> None:  # type: ignore[no-untyped-def]
        """An ablation of the reranker. Fusion order survives, scores are None."""
        results = retrieve(
            "Stevens Johnson syndrome",
            query_encoder=HashingEncoder(),
            cross_encoder=None,
            sparse_index=corpus,
            embedding_model=MODEL,
        )

        assert results
        assert all(result.rerank_score is None for result in results)

    def test_the_sparse_index_is_built_from_stored_chunks(self, corpus, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Written under tmp_path. Without the override this wrote a real index
        into the real DATA_DIR, from a database about to be dropped."""
        count, path = build_sparse_index(path=tmp_path / "sparse-rebuild")

        assert str(tmp_path) in path

        assert count == LabelChunk.objects.count()


class TestTheEvaluationHarnessEndToEnd:
    """The harness, run over synthetic judgements, proving it works unrun on real ones."""

    def _gold(self) -> GoldSet:
        lines = [
            '{"query_id": "q1", "query": "metformin lactic acidosis", '
            f'"relevant": ["{_sha(*SECTIONS[0])}"], "note": "synthetic"}}',
            '{"query_id": "q2", "query": "hepatic failure acetaminophen", '
            f'"relevant": ["{_sha(*SECTIONS[2])}"], "note": "synthetic"}}',
        ]
        return GoldSet(judgements=parse_judgements(lines), source=__file__)

    def _retriever(self, index):  # type: ignore[no-untyped-def]
        hashes = dict(LabelChunk.objects.values_list("id", "sha256"))

        def run(query: str) -> list[str]:
            return [
                hashes[result.chunk_id]
                for result in retrieve(
                    query,
                    query_encoder=HashingEncoder(),
                    cross_encoder=OverlapCrossEncoder(),
                    sparse_index=index,
                    embedding_model=MODEL,
                )
            ]

        return run

    def test_the_harness_scores_a_run_end_to_end(self, corpus) -> None:  # type: ignore[no-untyped-def]
        report = suite.run(self._gold(), self._retriever(corpus))

        assert report.queries == 2
        assert set(report.recall_at) == set(suite.DEFAULT_K_VALUES)
        assert 0.0 <= report.mrr <= 1.0

    def test_a_pipeline_that_leads_with_the_right_chunk_scores_perfectly(self, corpus) -> None:  # type: ignore[no-untyped-def]
        """Not a tautology: the retriever really has to rank these first."""
        report = suite.run(self._gold(), self._retriever(corpus))

        assert report.mrr == pytest.approx(1.0)
        assert report.recall_at[1] == pytest.approx(1.0)
        assert report.ndcg_at[1] == pytest.approx(1.0)

    def test_a_retriever_that_finds_nothing_scores_zero_rather_than_erroring(self, corpus) -> None:  # type: ignore[no-untyped-def]
        report = suite.run(self._gold(), lambda _query: [])

        assert report.mrr == 0.0
        assert report.recall_at[10] == 0.0

    def test_the_report_renders_and_serialises(self, corpus) -> None:  # type: ignore[no-untyped-def]
        report = suite.run(self._gold(), self._retriever(corpus))

        assert "MRR:" in suite.render(report)
        assert report.as_dict()["queries"] == 2
        assert len(report.as_dict()["per_query"]) == 2

    def test_an_empty_gold_set_is_refused_rather_than_averaged_over_nothing(self) -> None:
        with pytest.raises(RetrievalGoldSetError, match="empty gold set"):
            suite.run(GoldSet(judgements=(), source=__file__), lambda _query: [])


class TestResumption:
    """Interrupting a multi-hour embed must lose at most one batch.

    The work queue is the set of chunks with no vector for this model, re-derived
    every cycle. There is no cursor to lose, and the tests below are the evidence
    for the two claims that matter: it cannot skip a chunk and it cannot write
    one twice.
    """

    def _sections(self, count: int) -> None:
        document = LabelDocument.objects.create(set_id="resume-0000-4a11-9f00-000000000001")
        for ordinal in range(count):
            LabelSection.objects.create(
                document=document,
                section_code="warnings",
                ordinal=ordinal,
                text=f"Distinct finding number {ordinal} was observed in the trial.",
            )
        chunk_corpus()

    def test_a_bounded_run_embeds_only_its_limit(self) -> None:
        self._sections(6)

        run = embed_pending(HashingDocumentEncoder(), limit=2)

        assert run.embedded_this_run == 2
        assert run.remaining == 4

    def test_the_next_run_takes_exactly_what_the_first_left(self) -> None:
        """The whole of resumption: no skip, no repeat, no bookkeeping."""
        self._sections(6)
        first = embed_pending(HashingDocumentEncoder(), limit=2)
        second = embed_pending(HashingDocumentEncoder(), limit=2)

        assert (first.embedded_this_run, second.embedded_this_run) == (2, 2)
        assert second.already_embedded == 2
        assert second.remaining == 2
        assert ChunkEmbedding.objects.count() == 4

    def test_running_to_completion_leaves_nothing_pending(self) -> None:
        self._sections(5)

        run = embed_pending(HashingDocumentEncoder())

        assert run.remaining == 0
        assert ChunkEmbedding.objects.count() == LabelChunk.objects.count()

    def test_a_completed_corpus_re_run_does_nothing(self) -> None:
        """Idempotent: the second pass has no work, not duplicate work."""
        self._sections(4)
        embed_pending(HashingDocumentEncoder())

        again = embed_pending(HashingDocumentEncoder())

        assert again.embedded_this_run == 0
        assert ChunkEmbedding.objects.count() == LabelChunk.objects.count()

    def test_one_chunk_cannot_hold_two_vectors_for_one_model(self) -> None:
        """Enforced by the database, not by the code that happens to be running."""
        self._sections(3)
        embed_pending(HashingDocumentEncoder())
        embed_pending(HashingDocumentEncoder())

        counts = (
            ChunkEmbedding.objects.values("chunk_id", "model").annotate(rows=Count("id")).order_by()
        )

        assert all(row["rows"] == 1 for row in counts)

    def test_resumption_keys_on_the_model_so_a_second_encoder_is_separate_work(self) -> None:
        """An ablation encoder embeds the same chunks again rather than skipping."""
        self._sections(3)
        embed_pending(HashingDocumentEncoder())
        total = LabelChunk.objects.count()

        run = embed_pending(HashingDocumentEncoder(dimensions=768, model_id="stub/ablation"))

        assert run.embedded_this_run == total
        assert ChunkEmbedding.objects.count() == 2 * total

    def test_the_pending_query_is_the_work_queue(self) -> None:
        self._sections(4)

        assert len(pending_chunks("stub/hashing-document-encoder", 10)) == 4
        embed_pending(HashingDocumentEncoder(), limit=1)
        assert len(pending_chunks("stub/hashing-document-encoder", 10)) == 3

    def test_a_wrong_width_model_fails_before_writing_anything(self) -> None:
        """The assertion fires on the first batch, so hours are never spent."""
        self._sections(4)

        with pytest.raises(EmbeddingError, match="384-dimensional"):
            embed_pending(HashingDocumentEncoder(dimensions=384))

        assert ChunkEmbedding.objects.count() == 0


class TestTheArtifact:
    def _record(self, **overrides: object) -> dict:
        fields: dict = {
            "run_id": "20260907T000000Z",
            "params": {},
            "chunking": {},
            "embedding": {"model": MODEL},
            "sparse_path": Path("/nonexistent"),
            "seconds": 1.0,
        }
        fields.update(overrides)
        return artifact.collect(**fields).as_dict()

    def test_it_records_what_exists_rather_than_what_the_run_believed(self, corpus) -> None:  # type: ignore[no-untyped-def]
        document = self._record(embedding={"model": MODEL, "chunks_embedded_this_run": 4})

        assert document["corpus"]["label_sections"] == len(SECTIONS)
        assert document["corpus"]["chunks_total"] == LabelChunk.objects.count()
        assert document["dense"]["embeddings_total"] == ChunkEmbedding.objects.count()
        assert document["dense"]["embedding_dimensions"] == EMBEDDING_DIMENSIONS

    def test_it_measures_the_index_sizes_from_postgres(self, corpus) -> None:  # type: ignore[no-untyped-def]
        document = self._record()

        assert document["storage"]["chunk_embedding_bytes"] > 0
        assert document["storage"]["label_chunk_bytes"] > 0
        assert document["sparse"]["index_bytes"] == 0

    def test_it_states_that_no_accuracy_figure_is_in_it(self, corpus) -> None:  # type: ignore[no-untyped-def]
        """The artifact records what was built, not how well it retrieves."""
        document = self._record()

        assert document["quotable"]["withheld"] == ["retrieval accuracy"]
        assert "No gold set exists" in document["quotable"]["withheld_reason"]
        keys = {key.lower() for key in document["dense"]}
        assert not keys.intersection({"recall", "mrr", "ndcg"})

    def test_it_carries_the_commit_it_ran_at(self, corpus) -> None:  # type: ignore[no-untyped-def]
        assert self._record()["code_sha"]

    def test_it_writes_one_file_per_run(self, corpus, tmp_path) -> None:  # type: ignore[no-untyped-def]
        record = artifact.collect(
            run_id="20260907T000000Z",
            params={},
            chunking={},
            embedding={"model": MODEL},
            sparse_path=Path("/nonexistent"),
            seconds=1.0,
        )

        written = artifact.write(record, root=tmp_path)

        assert written.name == "index_20260907T000000Z.json"
        assert json.loads(written.read_text(encoding="utf-8"))["artifact"] == "index"

    def test_an_interrupted_run_is_visible_in_its_own_record(self) -> None:
        """chunks_without_embedding is what stops a partial run reading as finished."""
        document = LabelDocument.objects.create(set_id="partial-0000-4a11-9f00-000000000002")
        for ordinal in range(4):
            LabelSection.objects.create(
                document=document,
                section_code="warnings",
                ordinal=ordinal,
                text=f"Finding {ordinal} was observed during the study period.",
            )
        chunk_corpus()
        run = embed_pending(HashingDocumentEncoder(), limit=1)

        assert run.as_dict()["chunks_without_embedding"] == 3

    def test_the_width_it_reports_is_read_back_from_the_rows(self, corpus) -> None:  # type: ignore[no-untyped-def]
        """The declared constant and the measured width are separate claims."""
        document = self._record()

        assert document["dense"]["embedding_dimensions"] == EMBEDDING_DIMENSIONS
        assert document["dense"]["measured_dimensions"] == EMBEDDING_DIMENSIONS
        assert document["dense"]["recorded_dimensions"] == EMBEDDING_DIMENSIONS
        assert document["dense"]["rows_with_unexpected_dimensions"] == 0

    def test_it_says_how_much_of_the_corpus_has_no_vector_on_every_path(self) -> None:
        """A build that did not embed still has to report the shortfall.

        Taken from the database rather than from an embed run, so a chunk-only
        build cannot omit it and read as a finished index.
        """
        document = LabelDocument.objects.create(set_id="nodense-0000-4a11-9f00-000000000004")
        for ordinal in range(3):
            LabelSection.objects.create(
                document=document,
                section_code="warnings",
                ordinal=ordinal,
                text=f"Observation {ordinal} was recorded during follow up.",
            )
        chunk_corpus()

        record = self._record(chunking={"ran": True}, embedding={"model": MODEL, "ran": False})

        assert record["dense"]["ran"] is False
        assert record["dense"]["chunks_without_embedding"] == LabelChunk.objects.count()

    def test_an_embed_only_record_says_it_did_not_chunk(self, corpus) -> None:  # type: ignore[no-untyped-def]
        """ "This run did not chunk" and "nothing is chunked" are different facts."""
        record = self._record(chunking={"ran": False})

        assert record["corpus"]["ran"] is False
        assert record["corpus"]["chunks_total"] == LabelChunk.objects.count() > 0

    def test_it_carries_the_window_the_vectors_were_written_in(self, corpus) -> None:  # type: ignore[no-untyped-def]
        """Corpus-level and first-write-only; see the module docstring."""
        window = self._record()["dense"]["write_window"]

        assert window["first_vector_at"] is not None
        assert window["last_vector_at"] >= window["first_vector_at"]
        assert window["seconds"] >= 0

    def test_the_sparse_block_distinguishes_rebuilt_from_merely_present(
        self, corpus, tmp_path
    ) -> None:  # type: ignore[no-untyped-def]
        rows = list(LabelChunk.objects.order_by("id").values_list("id", "text"))
        sparse.build([i for i, _ in rows], [t for _, t in rows], path=tmp_path / "bm25")

        record = self._record(sparse_path=tmp_path / "bm25", sparse_rebuilt=False)

        assert record["sparse"]["chunks_indexed"] == len(rows)
        assert record["sparse"]["rebuilt"] is False

    def test_a_record_assembled_afterwards_admits_that_it_was(self, corpus, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """The block is a property of the entry point, not of an argument."""
        settings = get_settings()
        written = artifact.reconstruct(run_id="20260907T070358Z", settings=settings, root=tmp_path)
        document = json.loads(written.read_text(encoding="utf-8"))

        assert written.name == "index_20260907T070358Z.json"
        assert document["reconstructed"]["written_by"] == "signaldesk index artifact"
        assert "peak resident memory" in " ".join(document["reconstructed"]["never_captured"])
        assert "run_window" in " ".join(document["reconstructed"]["never_captured"])
        assert "run_window" not in document["dense"]

    def test_an_inline_record_never_claims_to_be_reconstructed(self, corpus) -> None:  # type: ignore[no-untyped-def]
        record = self._record()

        assert "reconstructed" not in record
        assert record["performance"]["peak_rss_bytes"] > 0

    def test_a_reconstructed_record_carries_no_performance_block(self, corpus, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """The only figures available to it would describe the wrong process.

        Listing peak memory as never captured while carrying a number under that
        name would have the file contradicting itself.
        """
        written = artifact.reconstruct(run_id="20260907T070358Z", root=tmp_path)
        document = json.loads(written.read_text(encoding="utf-8"))

        assert "performance" not in document
        assert "wall clock and peak memory" not in document["quotable"]["measured"]

    def test_it_names_the_weights_that_wrote_the_vectors(self, corpus) -> None:  # type: ignore[no-untyped-def]
        """A vector whose producing revision cannot be named is not traceable."""
        dense.store(
            list(LabelChunk.objects.values_list("id", flat=True)),
            embed_texts(
                HashingEncoder(),
                list(LabelChunk.objects.values_list("text", flat=True)),
                expected_dimensions=EMBEDDING_DIMENSIONS,
            ),
            model=MODEL,
            model_revision="abc123",
        )

        assert self._record()["dense"]["model_revisions_present"] == ["abc123"]

    def test_the_naming_evidence_is_measured_rather_than_asserted(self, corpus) -> None:  # type: ignore[no-untyped-def]
        """What the run id rests on has to be in the file, not in a report."""
        evidence = artifact.writer_evidence(MODEL, batch_rows=len(SECTIONS))

        assert evidence["writing_transactions"] >= 1
        assert evidence["out_of_order_transaction_boundaries"] == 0
        assert evidence["rows_ever_updated_or_deleted"] == 0
        assert evidence["unaccounted_xids"] == (
            evidence["xid_span"] - evidence["writing_transactions"]
        )
