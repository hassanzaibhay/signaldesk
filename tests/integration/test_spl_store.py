"""Writing labels, and re-writing them.

The pipeline is re-run whenever the scope widens, an override is added or the
cleaner changes, so the write has to converge on the same rows rather than
accumulate. A revised label with a shorter section is where an appending write
shows up first: the document would keep paragraphs no published label contains.
"""

from __future__ import annotations

import pytest

from signaldesk.ingest.spl.manifest import SOURCE, complete, decide, fail, start
from signaldesk.ingest.spl.parse import LabelRecord, LabelSectionRecord
from signaldesk.ingest.spl.store import attach_drug_keys, sections_for, store_labels
from signaldesk.web.documents.models import LabelDocument, LabelDrugKey, LabelSection
from signaldesk.web.signals.models import IngestManifest

pytestmark = [pytest.mark.integration, pytest.mark.django_db(transaction=True)]

SET_ID = "6a1b2c3d-0000-4a11-9f00-000000000001"


def _record(**overrides: object) -> LabelRecord:
    fields: dict[str, object] = {
        "set_id": SET_ID,
        "spl_id": "revision-1",
        "version": "3",
        "effective_time": "20240118",
        "brand_names": ["EXAMPLAMAB"],
        "rxcuis": ["1234567"],
        "sections": [
            LabelSectionRecord(section_code="boxed_warning", ordinal=0, text="WARNING: one"),
            LabelSectionRecord(section_code="adverse_reactions", ordinal=0, text="Headache."),
            LabelSectionRecord(section_code="adverse_reactions", ordinal=1, text="Nausea."),
        ],
    }
    fields.update(overrides)
    return LabelRecord(**fields)  # type: ignore[arg-type]


def _store(record: LabelRecord, folded: str = "EXAMPLAMAB 10MG") -> None:
    store_labels([record], folded_string=folded, query="EXAMPLAMAB", route="cleaned_string")


def test_one_pass_writes_the_document_its_sections_and_the_key() -> None:
    _store(_record())
    assert LabelDocument.objects.count() == 1
    assert LabelSection.objects.count() == 3
    assert LabelDrugKey.objects.count() == 1
    key = LabelDrugKey.objects.get()
    assert key.folded_string == "EXAMPLAMAB 10MG"
    assert key.route == "cleaned_string"
    assert key.ingredient_rxcui is None


def test_storing_the_same_label_twice_changes_nothing() -> None:
    _store(_record())
    _store(_record())
    assert LabelDocument.objects.count() == 1
    assert LabelSection.objects.count() == 3
    assert LabelDrugKey.objects.count() == 1


def test_a_revision_replaces_sections_rather_than_appending() -> None:
    """The shortened section is the case an appending write gets wrong."""
    _store(_record())
    _store(
        _record(
            spl_id="revision-2",
            version="4",
            effective_time="20250601",
            sections=[
                LabelSectionRecord(
                    section_code="adverse_reactions", ordinal=0, text="Headache only."
                )
            ],
        )
    )
    assert LabelDocument.objects.count() == 1
    document = LabelDocument.objects.get()
    assert document.spl_id == "revision-2"
    assert document.effective_time == "20250601"
    texts = list(LabelSection.objects.values_list("text", flat=True))
    assert texts == ["Headache only."]


def test_two_strings_can_reach_one_label() -> None:
    _store(_record(), folded="EXAMPLAMAB 10MG")
    _store(_record(), folded="EXAMPLAMAB 20MG")
    assert LabelDocument.objects.count() == 1
    assert LabelDrugKey.objects.count() == 2


def test_sections_for_is_the_retrieval_lookup() -> None:
    _store(_record())
    everything = sections_for("EXAMPLAMAB 10MG")
    assert len(everything) == 3
    boxed = sections_for("EXAMPLAMAB 10MG", codes=("boxed_warning",))
    assert [section.text for section in boxed] == ["WARNING: one"]
    assert sections_for("SOMETHING ELSE") == []


def test_deleting_a_document_takes_its_sections_and_keys_with_it() -> None:
    _store(_record())
    LabelDocument.objects.all().delete()
    assert LabelSection.objects.count() == 0
    assert LabelDrugKey.objects.count() == 0


class TestTheManifest:
    def test_an_unseen_unit_is_ingested(self) -> None:
        assert decide("str:abc").should_ingest

    def test_a_completed_unit_is_skipped(self) -> None:
        start("str:abc")
        complete(
            "str:abc",
            checksum="d" * 64,
            row_counts={"documents": 1},
            bytes_downloaded=10,
            page_cap=False,
        )
        decision = decide("str:abc")
        assert not decision.should_ingest
        assert decision.reason == "already completed"

    def test_a_failed_unit_is_retried(self) -> None:
        start("str:abc")
        fail("str:abc", "boom")
        assert decide("str:abc").should_ingest

    def test_force_overrides_a_completed_unit(self) -> None:
        start("str:abc")
        complete(
            "str:abc",
            checksum="d" * 64,
            row_counts={"documents": 1},
            bytes_downloaded=10,
            page_cap=False,
        )
        assert decide("str:abc", force=True).should_ingest

    def test_the_source_does_not_collide_with_faers(self) -> None:
        start("str:abc")
        assert IngestManifest.objects.filter(source=SOURCE).count() == 1
        assert IngestManifest.objects.filter(source="faers").count() == 0

    def test_the_unit_fits_the_column(self) -> None:
        unit = "str:" + "a" * 24
        start(unit)
        assert IngestManifest.objects.get(source=SOURCE, unit=unit).unit == unit


class TestThePageCapSurvivesTheFetch:
    """A unit that skips its fetch has to learn about truncation from the row.

    Otherwise it reports what it never measured. In the 06:05 artifact 190 rows
    said page_cap false without paging, three of them wrongly - ACETAMINOPHEN,
    IBUPROFEN and IBUPROFEN. all sit on queries that hit the ten-page cap.
    """

    def _complete(self, unit: str, *, page_cap: bool) -> None:
        start(unit)
        complete(
            unit,
            checksum="d" * 64,
            row_counts={"documents": 1000, "sections": 1364},
            bytes_downloaded=10,
            page_cap=page_cap,
        )

    def test_a_capped_fetch_is_readable_by_the_unit_that_skips(self) -> None:
        self._complete("str:capped", page_cap=True)
        assert decide("str:capped").page_capped is True

    def test_an_uncapped_fetch_is_false_not_none(self) -> None:
        self._complete("str:whole", page_cap=False)
        assert decide("str:whole").page_capped is False

    def test_a_row_written_before_the_flag_existed_is_none_not_false(self) -> None:
        """The distinction the artifact has to preserve.

        P12 excludes page-capped strings by reading this field. False is a
        measurement that truncation did not happen; None is the absence of one.
        Collapsing them would put unmeasured strings into a corpus that claims to
        exclude truncated ones.
        """
        start("str:legacy")
        IngestManifest.objects.filter(source=SOURCE, unit="str:legacy").update(
            status=IngestManifest.Status.COMPLETED,
            row_counts={"documents": 5, "sections": 9},
        )
        decision = decide("str:legacy")
        assert not decision.should_ingest
        assert decision.page_capped is None
        assert decision.page_capped is not False

    def test_the_flag_cannot_be_summed_as_a_row_count(self) -> None:
        """The encoding, pinned rather than remembered.

        Stored flat as 0/1 or as a bool, a future sum(row_counts.values()) would
        absorb the flag as one extra row and be quietly wrong. bool is an int, so
        a type check would not catch it either. Nested under a dict, that sum
        raises instead.
        """
        self._complete("str:capped", page_cap=True)
        row = IngestManifest.objects.get(source=SOURCE, unit="str:capped")

        with pytest.raises(TypeError):
            sum(row.row_counts.values())

        # row_count is still the row-bearing keys and nothing else.
        assert row.row_count == 1000 + 1364


class TestAttachingKeysToAnotherStringsFetch:
    """Several strings clean to one query, so only the first fetches.

    The rest must still reach the labels: the FAERS string is the join key back
    to the signal table, and a lookup on the exact string has to resolve.
    """

    def test_a_duplicate_string_reaches_the_same_documents(self) -> None:
        _store(_record(), folded="GABAPENTIN")
        written = attach_drug_keys(
            folded_string="GABAPENTIN.", query="EXAMPLAMAB", route="cleaned_string"
        )
        assert written == 1
        reached = set(
            LabelDrugKey.objects.filter(folded_string="GABAPENTIN.").values_list(
                "document__set_id", flat=True
            )
        )
        assert reached == {SET_ID}
        assert sections_for("GABAPENTIN.")

    def test_it_writes_no_documents_and_no_sections(self) -> None:
        """The constraint: those belong to the unit that fetched them.

        Re-storing here is what would put distinct_documents and section_writes
        wrong again, in a new way.
        """
        _store(_record(), folded="GABAPENTIN")
        before = (LabelDocument.objects.count(), LabelSection.objects.count())
        attach_drug_keys(folded_string="GABAPENTIN.", query="EXAMPLAMAB", route="cleaned_string")
        after = (LabelDocument.objects.count(), LabelSection.objects.count())
        assert before == after
        assert LabelDrugKey.objects.count() == 2

    def test_an_unknown_query_attaches_nothing(self) -> None:
        """The state the pipeline reads as an inconsistency when the manifest
        claims documents. Here nothing was ever stored, so zero is correct."""
        assert (
            attach_drug_keys(
                folded_string="ZOFRAN", query="NOTHING EVER FETCHED THIS", route="cleaned_string"
            )
            == 0
        )
        assert not LabelDrugKey.objects.filter(folded_string="ZOFRAN").exists()

    def test_it_converges_rather_than_accumulating(self) -> None:
        """Re-running the pipeline must not multiply the reach-through rows."""
        _store(_record(), folded="GABAPENTIN")
        written = [
            attach_drug_keys(
                folded_string="GABAPENTIN.", query="EXAMPLAMAB", route="cleaned_string"
            )
            for _ in range(3)
        ]
        assert LabelDrugKey.objects.filter(folded_string="GABAPENTIN.").count() == 1
        # The returned count converges too. It did not before: after the first
        # attach two strings carried this query, and the ordering-driven DISTINCT
        # then returned two rows for one document on every later call. The row
        # count was right and the number reported to the artifact was double.
        assert written == [1, 1, 1]

    def test_the_second_attacher_counts_documents_not_document_string_pairs(self) -> None:
        """The defect needs three members and more than one document to show.

        A two-member group cannot catch it: the only attacher sees one string
        already keyed to the query, so its multiplier is one and the wrong answer
        equals the right one. The real scope has exactly one three-way group,
        RITUXIMAB / RITUXIMAB. / RITUXIMAB (UNKNOWN), which is what this is.

        Two documents, so a multiplier is visible as a multiple rather than as an
        off-by-one that could be anything.
        """
        second = _record(set_id="6a1b2c3d-0000-4a11-9f00-000000000002")
        store_labels(
            [_record(), second],
            folded_string="RITUXIMAB",
            query="RITUXIMAB",
            route="cleaned_string",
        )

        first_attach = attach_drug_keys(
            folded_string="RITUXIMAB.", query="RITUXIMAB", route="cleaned_string"
        )
        second_attach = attach_drug_keys(
            folded_string="RITUXIMAB (UNKNOWN)", query="RITUXIMAB", route="cleaned_string"
        )

        assert first_attach == 2
        # Two strings now carry the query, so the unfixed lookup returned four.
        assert second_attach == 2
        assert LabelDrugKey.objects.filter(folded_string="RITUXIMAB (UNKNOWN)").count() == 2

    def test_the_reported_count_is_the_rows_written(self) -> None:
        """The invariant the artifact depends on, stated directly.

        drug_keys in a unit row is read as 'documents this string reaches'. It is
        only that if it equals the rows the attach wrote.
        """
        store_labels(
            [_record(), _record(set_id="6a1b2c3d-0000-4a11-9f00-000000000003")],
            folded_string="RITUXIMAB",
            query="RITUXIMAB",
            route="cleaned_string",
        )
        attach_drug_keys(folded_string="RITUXIMAB.", query="RITUXIMAB", route="cleaned_string")
        reported = attach_drug_keys(
            folded_string="RITUXIMAB (UNKNOWN)", query="RITUXIMAB", route="cleaned_string"
        )
        assert reported == LabelDrugKey.objects.filter(folded_string="RITUXIMAB (UNKNOWN)").count()

    def test_the_route_and_rxcui_are_recorded_on_the_attached_row(self) -> None:
        _store(_record(), folded="GABAPENTIN")
        attach_drug_keys(
            folded_string="GABAPENTIN.",
            query="EXAMPLAMAB",
            route="ingredient",
            ingredient_rxcui=83367,
        )
        row = LabelDrugKey.objects.get(folded_string="GABAPENTIN.")
        assert row.route == "ingredient"
        assert row.ingredient_rxcui == 83367
        assert row.query == "EXAMPLAMAB"
