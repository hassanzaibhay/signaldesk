"""The frame, and the page_cap null rule F4 deferred to P12."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from signaldesk.core.errors import AnnotationError
from signaldesk.evals.labeledness.frame import (
    CapState,
    DocumentRow,
    FetchUnit,
    ReachRow,
    build_frame,
    page_cap_state,
    read_fetch_units,
    string_cap_states,
)

pytestmark = pytest.mark.unit


def _document(document_id: int, *, primary: bool = True) -> DocumentRow:
    return DocumentRow(
        document_id=document_id, set_id=f"set-{document_id}", has_primary_section=primary
    )


class TestPageCapIsThreeStated:
    """F4 left page_cap null on rows that never measured it.

    The deferred half was that P12 must treat null as unknown-and-excluded rather
    than false, and that nothing in F4 forced it to. These are the tests that
    force it.

    Read this before trusting them: on the corpus as it stands the rule is
    NON-BINDING. All 160 live queries and all 200 artifact units carry true or
    false, and the 32 manifest rows holding null are orphans outside the current
    scope. The rule binds on the next unforced ingest, where the attach path can
    hand an attaching unit a null it must not read as false. So every fixture
    here is constructed, deliberately: a test drawn from today's data would pass
    without ever exercising the branch.
    """

    def test_a_null_page_cap_is_unknown_and_is_not_false(self) -> None:
        assert page_cap_state(None) is CapState.UNKNOWN
        assert page_cap_state(None) is not CapState.CLEAN

    def test_true_and_false_are_the_measurements_they_are(self) -> None:
        assert page_cap_state(True) is CapState.CAPPED
        assert page_cap_state(False) is CapState.CLEAN

    @pytest.mark.parametrize("value", [0, 1, "", "false", "true", [], {}])
    def test_a_non_boolean_is_rejected_rather_than_coerced(self, value: object) -> None:
        """Every one of these is falsey or truthy and none is a measurement.

        ``bool(0)`` and ``bool("")`` both read as "not capped", which is the
        silent widening this guards. ``bool`` is checked before ``int`` because
        ``bool`` is an ``int``; without that ordering ``True`` would fall through
        to the rejection branch.
        """
        with pytest.raises(AnnotationError, match="page_cap must be true, false or null"):
            page_cap_state(value)

    def test_a_null_page_cap_row_is_excluded_from_the_frame(self) -> None:
        """The property, stated end to end rather than on the helper.

        A string whose fetch never measured truncation must not reach the frame,
        and the reason recorded must say unknown rather than capped: the two
        exclude alike but a reader of the manifest has to be able to tell "we
        measured a cap" from "we measured nothing".
        """
        frame = build_frame(
            documents=[_document(1), _document(2)],
            reach=[
                ReachRow(folded_string="CLEANDRUG", query="CLEANDRUG", document_id=1),
                ReachRow(folded_string="UNMEASURED", query="UNMEASURED", document_id=2),
            ],
            units=[
                FetchUnit(query="CLEANDRUG", folded_string="CLEANDRUG", state=CapState.CLEAN),
                FetchUnit(query="UNMEASURED", folded_string="UNMEASURED", state=CapState.UNKNOWN),
            ],
        )

        assert "UNMEASURED" not in frame.documents_by_string
        assert frame.excluded_strings["UNMEASURED"] is CapState.UNKNOWN
        assert frame.excluded_strings["UNMEASURED"] is not CapState.CLEAN
        assert frame.counts.strings_unknown == 1
        assert frame.counts.strings_capped == 0
        assert frame.counts.strings_eligible == 1

    def test_a_string_with_no_manifest_row_at_all_is_unknown_not_clean(self) -> None:
        """The default, which is where a widening would most plausibly land.

        A string the artifact never mentions has had nothing measured about it.
        Defaulting that to CLEAN would admit a truncated fetch on the strength of
        the artifact being incomplete.
        """
        frame = build_frame(
            documents=[_document(1)],
            reach=[ReachRow(folded_string="ABSENT", query="ABSENT", document_id=1)],
            units=[],
        )
        assert frame.excluded_strings["ABSENT"] is CapState.UNKNOWN
        assert frame.counts.strings_eligible == 0

    def test_an_artifact_with_a_missing_page_cap_field_reads_as_unknown(
        self, tmp_path: Path
    ) -> None:
        """Absent and explicitly null are the same statement: nothing measured."""
        artifact = tmp_path / "spl_ingest.json"
        artifact.write_text(
            json.dumps(
                {
                    "units": [
                        {"folded_string": "A", "query": "A", "page_cap": False},
                        {"folded_string": "B", "query": "B", "page_cap": None},
                        {"folded_string": "C", "query": "C"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        states = {unit.folded_string: unit.state for unit in read_fetch_units(artifact)}
        assert states == {"A": CapState.CLEAN, "B": CapState.UNKNOWN, "C": CapState.UNKNOWN}


class TestCapStateAcrossSeveralQueries:
    def test_a_capped_query_anywhere_excludes_the_string(self) -> None:
        states = string_cap_states(
            [
                FetchUnit(query="q1", folded_string="DRUG", state=CapState.CLEAN),
                FetchUnit(query="q2", folded_string="DRUG", state=CapState.CAPPED),
            ]
        )
        assert states["DRUG"] is CapState.CAPPED

    def test_capped_outranks_unknown_in_the_reported_reason(self) -> None:
        """Both exclude. The ordering is about what the manifest tells a reader."""
        states = string_cap_states(
            [
                FetchUnit(query="q1", folded_string="DRUG", state=CapState.UNKNOWN),
                FetchUnit(query="q2", folded_string="DRUG", state=CapState.CAPPED),
            ]
        )
        assert states["DRUG"] is CapState.CAPPED


class TestTheTwoExclusions:
    def test_the_page_cap_exclusion_is_at_the_string_level(self) -> None:
        """A document reached from both a capped and a clean string stays.

        The mechanism is that a capped fetch truncates a drug's document set, not
        that it damages a document. Excluding the shared document would drop a
        document whose coverage is complete by another route -- 42 of them on the
        real corpus.
        """
        frame = build_frame(
            documents=[_document(1), _document(2)],
            reach=[
                ReachRow(folded_string="CAPPED", query="CAPPED", document_id=1),
                ReachRow(folded_string="CAPPED", query="CAPPED", document_id=2),
                ReachRow(folded_string="CLEAN", query="CLEAN", document_id=2),
            ],
            units=[
                FetchUnit(query="CAPPED", folded_string="CAPPED", state=CapState.CAPPED),
                FetchUnit(query="CLEAN", folded_string="CLEAN", state=CapState.CLEAN),
            ],
        )
        assert frame.documents_by_string == {"CLEAN": (2,)}
        assert frame.counts.documents_eligible == 1

    def test_a_document_without_adverse_reactions_text_is_out_of_frame(self) -> None:
        frame = build_frame(
            documents=[_document(1, primary=False), _document(2)],
            reach=[
                ReachRow(folded_string="DRUG", query="DRUG", document_id=1),
                ReachRow(folded_string="DRUG", query="DRUG", document_id=2),
            ],
            units=[FetchUnit(query="DRUG", folded_string="DRUG", state=CapState.CLEAN)],
        )
        assert frame.documents_by_string["DRUG"] == (2,)
        assert frame.counts.documents_without_primary_section == 1

    def test_a_clean_string_with_no_usable_document_is_not_counted_as_capped(self) -> None:
        """Two different reasons to be out of frame, kept apart in the counts."""
        frame = build_frame(
            documents=[_document(1, primary=False)],
            reach=[ReachRow(folded_string="DRUG", query="DRUG", document_id=1)],
            units=[FetchUnit(query="DRUG", folded_string="DRUG", state=CapState.CLEAN)],
        )
        assert frame.counts.strings_clean == 1
        assert frame.counts.strings_capped == 0
        assert frame.counts.strings_unknown == 0
        assert frame.counts.strings_eligible == 0

    def test_sibling_strings_share_a_query_group(self) -> None:
        frame = build_frame(
            documents=[_document(1)],
            reach=[
                ReachRow(folded_string="PREDNISONE", query="PREDNISONE", document_id=1),
                ReachRow(folded_string="PREDNISONE.", query="PREDNISONE", document_id=1),
            ],
            units=[
                FetchUnit(query="PREDNISONE", folded_string="PREDNISONE", state=CapState.CLEAN),
                FetchUnit(query="PREDNISONE", folded_string="PREDNISONE.", state=CapState.CLEAN),
            ],
        )
        assert frame.strings_by_query == {"PREDNISONE": ("PREDNISONE", "PREDNISONE.")}
        assert frame.counts.query_groups_eligible == 1
        assert frame.counts.strings_eligible == 2
        assert frame.documents_for_query("PREDNISONE") == (1,)


class TestAMalformedArtifactStopsTheDraw:
    """The draw is one-shot, so a half-readable artifact is not a thing to salvage."""

    def test_a_missing_artifact_is_named(self, tmp_path: Path) -> None:
        with pytest.raises(AnnotationError, match="cannot read SPL ingest artifact"):
            read_fetch_units(tmp_path / "absent.json")

    def test_a_file_without_a_units_list_is_rejected(self, tmp_path: Path) -> None:
        artifact = tmp_path / "wrong.json"
        artifact.write_text(json.dumps({"run_id": "x"}), encoding="utf-8")
        with pytest.raises(AnnotationError, match="not an SPL ingest artifact"):
            read_fetch_units(artifact)

    def test_a_unit_that_is_not_an_object_is_rejected(self, tmp_path: Path) -> None:
        artifact = tmp_path / "bad.json"
        artifact.write_text(json.dumps({"units": ["ACETAMINOPHEN"]}), encoding="utf-8")
        with pytest.raises(AnnotationError, match=r"units\[0\] is not an object"):
            read_fetch_units(artifact)

    def test_a_unit_without_a_string_and_query_is_rejected(self, tmp_path: Path) -> None:
        artifact = tmp_path / "bad.json"
        artifact.write_text(json.dumps({"units": [{"page_cap": False}]}), encoding="utf-8")
        with pytest.raises(AnnotationError, match="no folded_string/query pair"):
            read_fetch_units(artifact)

    def test_a_non_boolean_page_cap_in_a_real_artifact_shape_is_rejected(
        self, tmp_path: Path
    ) -> None:
        """The rejection reaches the caller through the artifact reader, not only
        through the helper, so a draw cannot start on a malformed flag."""
        artifact = tmp_path / "bad.json"
        artifact.write_text(
            json.dumps({"units": [{"folded_string": "A", "query": "A", "page_cap": 0}]}),
            encoding="utf-8",
        )
        with pytest.raises(AnnotationError, match="page_cap must be true, false or null"):
            read_fetch_units(artifact)
