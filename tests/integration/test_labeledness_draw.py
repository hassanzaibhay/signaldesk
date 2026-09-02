"""The draw, end to end, against a real database and a real signal partition.

The unit tests cover the pure half. This covers the half that reads Postgres and
the signal parquet, and it exists because the draw is one-shot: it runs once,
against a corpus that will have moved by the time anyone notices a mistake, and
its failure mode is a silently wrong sample rather than an error. Nobody gets to
re-run it after a day of annotation has been spent on its output.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from signaldesk.core.config import Settings
from signaldesk.evals.labeledness.draw import draw, load_documents_and_reach, load_sections
from signaldesk.evals.labeledness.frame import CapState
from signaldesk.evals.labeledness.manifest import read_manifest
from signaldesk.evals.labeledness.schedule import (
    TOTAL_REPEATS,
    TOTAL_SCREENS,
    TOTAL_UNIQUE,
)
from signaldesk.web.documents.models import LabelDocument, LabelDrugKey, LabelSection

pytestmark = [pytest.mark.integration, pytest.mark.django_db(transaction=True)]

RUN_ID = "20260902T000000Z"

#: Enough distinct events that 300 pairs can be drawn without exhausting the
#: frame, over few enough drugs that the twin and exclusion cases all appear.
EVENTS = [f"EVENT {index:03d}" for index in range(120)]

#: query -> (folded strings, page_cap as it appears in the artifact)
#:
#: PREDNISONE carries a twin, so the sibling presentation has something to use.
#: IBUPROFEN is page-capped and ORPHANDRUG has never had truncation measured;
#: both must leave the frame, and for different recorded reasons.
UNITS: dict[str, tuple[tuple[str, ...], object]] = {
    "PREDNISONE": (("PREDNISONE", "PREDNISONE."), False),
    "METHOTREXATE": (("METHOTREXATE",), False),
    "HUMIRA": (("HUMIRA",), False),
    "ENBREL": (("ENBREL",), False),
    "IBUPROFEN": (("IBUPROFEN",), True),
    "ORPHANDRUG": (("ORPHANDRUG",), None),
}


def _seed_corpus() -> None:
    """Three documents per query, one of which carries no adverse-reactions text."""
    for query, (strings, _cap) in UNITS.items():
        for index in range(3):
            document = LabelDocument.objects.create(
                set_id=f"{query.lower()}-set-{index}", spl_id=f"{query}-{index}", version="1"
            )
            if index == 2:
                # No adverse-reactions text at all: out of frame by the predicate,
                # including the case where the document carries other sections.
                LabelSection.objects.create(
                    document=document, section_code="warnings", ordinal=0, text="Warnings only."
                )
            else:
                LabelSection.objects.create(
                    document=document,
                    section_code="adverse_reactions",
                    ordinal=0,
                    text=f"{query} caused EVENT {index:03d} in some patients. Micro sign 5 \u00b5g.",
                )
                LabelSection.objects.create(
                    document=document,
                    section_code="boxed_warning",
                    ordinal=0,
                    text=f"WARNING: {query} is associated with a class effect.",
                )
            for folded in strings:
                LabelDrugKey.objects.create(
                    folded_string=folded,
                    query=query,
                    route="cleaned_string",
                    document=document,
                )


def _write_signal_partition(root: Path) -> None:
    rows = []
    for strings, _cap in UNITS.values():
        for folded in strings:
            for position, event in enumerate(EVENTS):
                rows.append(
                    {
                        "run_id": RUN_ID,
                        "drug": folded,
                        "pt": event,
                        # Two pairs per drug sit below the minimum count and must
                        # not reach the frame.
                        "a": 2 if position < 2 else 3 + position,
                        "flag_ror_prr_bcpnn": position % 2 == 0,
                    }
                )
    partition = root / "parquet" / "signal" / f"run={RUN_ID}"
    partition.mkdir(parents=True)
    pl.DataFrame(rows).write_parquet(partition / "part-0.parquet")


def _write_spl_artifact(path: Path) -> None:
    units = []
    for query, (strings, cap) in UNITS.items():
        for folded in strings:
            units.append({"folded_string": folded, "query": query, "page_cap": cap})
    path.write_text(json.dumps({"run_id": "x", "units": units}), encoding="utf-8")


@pytest.fixture
def drawn(tmp_path: Path) -> tuple[Path, Settings]:
    _seed_corpus()
    settings = Settings(
        _env_file=None,
        django_secret_key="test-only-not-a-secret",
        data_dir=tmp_path / "data",
        cache_dir=tmp_path / "cache",
        model_dir=tmp_path / "models",
    )
    _write_signal_partition(tmp_path / "data")
    artifact = tmp_path / "spl_ingest_test.json"
    _write_spl_artifact(artifact)
    history = tmp_path / "history"
    result = draw(
        seed=20260902,
        guideline_version="v1",
        spl_artifact=artifact,
        signal_artifact="signals_test.json",
        history_dir=history,
        settings=settings,
        run_id=RUN_ID,
    )
    return result.path, settings


class TestTheCorpusReadersReturnWhatTheFrameNeeds:
    def test_documents_report_whether_they_carry_adverse_reactions_text(self) -> None:
        _seed_corpus()
        documents, reach = load_documents_and_reach()
        assert len(documents) == len(UNITS) * 3
        assert sum(1 for row in documents if row.has_primary_section) == len(UNITS) * 2
        assert len(reach) == sum(len(strings) for strings, _ in UNITS.values()) * 3

    def test_sections_come_back_in_guideline_search_order(self) -> None:
        _seed_corpus()
        document = LabelDocument.objects.get(set_id="humira-set-0")
        blocks = load_sections([document.id])[document.id]
        assert [code for code, _, _ in blocks] == ["adverse_reactions", "boxed_warning"]


class TestTheDrawWritesACommittableManifest:
    def test_it_writes_the_full_schedule_and_reloads(self, drawn: tuple[Path, Settings]) -> None:
        path, _settings = drawn
        manifest = read_manifest(path)
        assert len(manifest.screens) == TOTAL_SCREENS
        assert sum(1 for screen in manifest.screens if screen.is_repeat) == TOTAL_REPEATS
        assert manifest.seed == 20260902
        assert manifest.guideline_version == "v1"
        assert manifest.signal_run_id == RUN_ID
        assert manifest.signal_partitions_on_disk == 1

    def test_the_artifact_on_disk_is_pure_ascii(self, drawn: tuple[Path, Settings]) -> None:
        """The corpus fixture contains a micro sign, so this is not vacuous."""
        path, _settings = drawn
        raw = path.read_bytes()
        assert all(byte < 0x80 for byte in raw)
        assert rb"\u00b5" in raw

    def test_every_screen_carries_the_text_it_will_be_judged_against(
        self, drawn: tuple[Path, Settings]
    ) -> None:
        """The reason the harness needs no database.

        Every byte the annotator sees is in the manifest, digested, so the
        stimulus cannot move under a verdict and a verdict stays re-checkable.
        """
        path, _settings = drawn
        manifest = read_manifest(path)
        for screen in manifest.screens:
            assert screen.primary_chars > 0
            assert "adverse_reactions" in screen.section_codes
            assert screen.digests()

    def test_the_same_seed_reproduces_the_same_draw(self, tmp_path: Path) -> None:
        _seed_corpus()
        settings = Settings(
            _env_file=None,
            django_secret_key="test-only-not-a-secret",
            data_dir=tmp_path / "data",
            cache_dir=tmp_path / "cache",
            model_dir=tmp_path / "models",
        )
        _write_signal_partition(tmp_path / "data")
        artifact = tmp_path / "spl.json"
        _write_spl_artifact(artifact)

        def _run(seed: int, name: str) -> list[tuple[str, str, str]]:
            result = draw(
                seed=seed,
                guideline_version="v1",
                spl_artifact=artifact,
                signal_artifact="signals_test.json",
                history_dir=tmp_path / name,
                settings=settings,
                run_id=RUN_ID,
            )
            return [
                (screen.pair_id, screen.drug_string, screen.set_id)
                for screen in result.manifest.screens
            ]

        assert _run(20260902, "a") == _run(20260902, "b")
        assert _run(20260902, "a") != _run(20260903, "c")


class TestBothExclusionsHoldOnRealRows:
    def test_a_page_capped_string_never_reaches_a_screen(
        self, drawn: tuple[Path, Settings]
    ) -> None:
        path, _settings = drawn
        manifest = read_manifest(path)
        drawn_strings = {screen.drug_string for screen in manifest.screens}
        assert "IBUPROFEN" not in drawn_strings
        assert manifest.frame.strings_capped == 1

    def test_a_string_whose_page_cap_was_never_measured_never_reaches_a_screen(
        self, drawn: tuple[Path, Settings]
    ) -> None:
        """The F4 deferral, on real rows rather than a constructed CapState.

        The unit test proves null maps to UNKNOWN. This proves the UNKNOWN
        actually removes the string from a drawn manifest, which is the property
        that matters and the one nothing in F4 forced.
        """
        path, _settings = drawn
        manifest = read_manifest(path)
        assert "ORPHANDRUG" not in {screen.drug_string for screen in manifest.screens}
        assert manifest.frame.strings_unknown == 1
        assert manifest.frame.strings_capped == 1

    def test_a_document_without_adverse_reactions_text_is_never_shown(
        self, drawn: tuple[Path, Settings]
    ) -> None:
        path, _settings = drawn
        manifest = read_manifest(path)
        empty = {
            document.set_id
            for document in LabelDocument.objects.all()
            if not document.sections.filter(section_code="adverse_reactions").exists()
        }
        assert empty
        assert empty.isdisjoint({screen.set_id for screen in manifest.screens})
        assert manifest.frame.documents_without_primary_section == len(UNITS)

    def test_pairs_below_the_minimum_count_are_out_of_frame(
        self, drawn: tuple[Path, Settings]
    ) -> None:
        path, _settings = drawn
        manifest = read_manifest(path)
        eligible_strings = len(UNITS["PREDNISONE"][0]) + 3
        assert manifest.frame.strings_eligible == eligible_strings
        # 120 events per drug, two of them below a >= 3, over 4 query groups.
        assert manifest.frame_pairs == 4 * (len(EVENTS) - 2)


class TestDeduplicationAndSiblings:
    def test_a_twin_pair_is_drawn_once_not_twice(self, drawn: tuple[Path, Settings]) -> None:
        """PREDNISONE and PREDNISONE. are one question, not two.

        Left undeduplicated they would double the sampling mass of every group
        that carries a twin.
        """
        path, _settings = drawn
        manifest = read_manifest(path)
        firsts = [screen for screen in manifest.screens if not screen.is_repeat]
        assert len({screen.pair_id for screen in firsts}) == TOTAL_UNIQUE
        for screen in firsts:
            assert screen.pair_id.startswith(screen.query + "||")

    def test_a_repeat_uses_the_sibling_string_where_the_group_has_one(
        self, drawn: tuple[Path, Settings]
    ) -> None:
        path, _settings = drawn
        manifest = read_manifest(path)
        by_id = {screen.screen_id: screen for screen in manifest.screens}
        siblings = 0
        for screen in manifest.screens:
            if not screen.is_repeat:
                continue
            assert screen.repeat_of is not None
            original = by_id[screen.repeat_of]
            assert original.pair_id == screen.pair_id
            assert original.set_id == screen.set_id
            assert original.digests() == screen.digests()
            if screen.query == "PREDNISONE":
                assert screen.drug_string == "PREDNISONE."
                assert original.drug_string == "PREDNISONE"
                siblings += 1
        assert siblings > 0


class TestTheReserveIsPreRegistered:
    def test_it_is_committed_with_the_primary_draw_and_disjoint_from_it(
        self, drawn: tuple[Path, Settings]
    ) -> None:
        path, _settings = drawn
        manifest = read_manifest(path)
        assert manifest.reserve
        primary = {screen.pair_id for screen in manifest.screens}
        assert primary.isdisjoint({screen.pair_id for screen in manifest.reserve})

    def test_the_artifact_says_an_unannotated_reserve_is_not_an_omission(
        self, drawn: tuple[Path, Settings]
    ) -> None:
        """So nobody later reads the expected state as a gap in the work."""
        path, _settings = drawn
        manifest = read_manifest(path)
        assert "not an omission" in manifest.reserve_note
        assert "insurance" in manifest.reserve_note

    def test_the_notes_record_that_the_null_rule_is_non_binding_here(
        self, drawn: tuple[Path, Settings]
    ) -> None:
        path, _settings = drawn
        manifest = read_manifest(path)
        assert any("null is excluded as unknown" in note for note in manifest.notes)


class TestTheDrawRefusesToRunOnNothing:
    def test_no_signal_partition_stops_it(self, tmp_path: Path) -> None:
        from signaldesk.core.errors import AnnotationError

        settings = Settings(
            _env_file=None,
            django_secret_key="test-only-not-a-secret",
            data_dir=tmp_path / "data",
            cache_dir=tmp_path / "cache",
            model_dir=tmp_path / "models",
        )
        artifact = tmp_path / "spl.json"
        _write_spl_artifact(artifact)
        with pytest.raises(AnnotationError, match="no signal run partition"):
            draw(
                seed=1,
                guideline_version="v1",
                spl_artifact=artifact,
                signal_artifact="signals_test.json",
                history_dir=tmp_path / "history",
                settings=settings,
            )

    def test_a_frame_too_small_to_draw_from_stops_it(self, tmp_path: Path) -> None:
        from signaldesk.core.errors import AnnotationError

        settings = Settings(
            _env_file=None,
            django_secret_key="test-only-not-a-secret",
            data_dir=tmp_path / "data",
            cache_dir=tmp_path / "cache",
            model_dir=tmp_path / "models",
        )
        partition = tmp_path / "data" / "parquet" / "signal" / f"run={RUN_ID}"
        partition.mkdir(parents=True)
        pl.DataFrame(
            [{"run_id": RUN_ID, "drug": "PREDNISONE", "pt": "EVENT 000", "a": 9}]
        ).write_parquet(partition / "part-0.parquet")
        _seed_corpus()
        artifact = tmp_path / "spl.json"
        _write_spl_artifact(artifact)
        with pytest.raises(AnnotationError, match="fewer than the 300"):
            draw(
                seed=1,
                guideline_version="v1",
                spl_artifact=artifact,
                signal_artifact="signals_test.json",
                history_dir=tmp_path / "history",
                settings=settings,
            )


class TestTheFrameReportsExclusionReasons:
    def test_each_excluded_string_carries_why(self, tmp_path: Path) -> None:
        from signaldesk.evals.labeledness.frame import build_frame, read_fetch_units

        _seed_corpus()
        documents, reach = load_documents_and_reach()
        artifact = tmp_path / "spl_reasons.json"
        _write_spl_artifact(artifact)
        frame = build_frame(documents=documents, reach=reach, units=read_fetch_units(artifact))
        assert frame.excluded_strings["IBUPROFEN"] is CapState.CAPPED
        assert frame.excluded_strings["ORPHANDRUG"] is CapState.UNKNOWN
        assert frame.excluded_strings["ORPHANDRUG"] is not CapState.CLEAN
