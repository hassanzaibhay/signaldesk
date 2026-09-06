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

import pytest
from tests.conftest import HashingEncoder, OverlapCrossEncoder

from signaldesk.core.config import get_settings
from signaldesk.evals.retrieval import suite
from signaldesk.evals.retrieval.gold import GoldSet, RetrievalGoldSetError, parse_judgements
from signaldesk.rag import chunking
from signaldesk.rag.embed import embed_texts
from signaldesk.rag.index import dense, sparse
from signaldesk.rag.index.corpus import build_sparse_index, chunk_corpus
from signaldesk.rag.retrieve import retrieve
from signaldesk.web.documents.models import (
    EMBEDDING_DIMENSIONS,
    LabelChunk,
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
            encoder=HashingEncoder(),
            cross_encoder=OverlapCrossEncoder(),
            sparse_index=corpus,
            embedding_model=MODEL,
        )

        assert results[0].section_code == "boxed_warning"
        assert "Lactic acidosis" in results[0].text

    def test_results_carry_where_each_retriever_found_them(self, corpus) -> None:  # type: ignore[no-untyped-def]
        results = retrieve(
            "hepatic failure acetaminophen",
            encoder=HashingEncoder(),
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
            encoder=HashingEncoder(),
            cross_encoder=OverlapCrossEncoder(),
            sparse_index=corpus,
            embedding_model=MODEL,
        )

        assert len(results) <= get_settings().rerank_top_k

    def test_omitting_the_cross_encoder_is_a_configuration_not_a_failure(self, corpus) -> None:  # type: ignore[no-untyped-def]
        """An ablation of the reranker. Fusion order survives, scores are None."""
        results = retrieve(
            "Stevens Johnson syndrome",
            encoder=HashingEncoder(),
            cross_encoder=None,
            sparse_index=corpus,
            embedding_model=MODEL,
        )

        assert results
        assert all(result.rerank_score is None for result in results)

    def test_the_sparse_index_is_built_from_stored_chunks(self, corpus) -> None:  # type: ignore[no-untyped-def]
        count, _path = build_sparse_index()

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
                    encoder=HashingEncoder(),
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
