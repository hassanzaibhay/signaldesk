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
from signaldesk.ingest.spl.store import sections_for, store_labels
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
        complete("str:abc", checksum="d" * 64, row_counts={"documents": 1}, bytes_downloaded=10)
        decision = decide("str:abc")
        assert not decision.should_ingest
        assert decision.reason == "already completed"

    def test_a_failed_unit_is_retried(self) -> None:
        start("str:abc")
        fail("str:abc", "boom")
        assert decide("str:abc").should_ingest

    def test_force_overrides_a_completed_unit(self) -> None:
        start("str:abc")
        complete("str:abc", checksum="d" * 64, row_counts={"documents": 1}, bytes_downloaded=10)
        assert decide("str:abc", force=True).should_ingest

    def test_the_source_does_not_collide_with_faers(self) -> None:
        start("str:abc")
        assert IngestManifest.objects.filter(source=SOURCE).count() == 1
        assert IngestManifest.objects.filter(source="faers").count() == 0

    def test_the_unit_fits_the_column(self) -> None:
        unit = "str:" + "a" * 24
        start(unit)
        assert IngestManifest.objects.get(source=SOURCE, unit=unit).unit == unit
