"""The append-only store: crash safety, tombstone undo, resume."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from signaldesk.core.errors import AnnotationError
from signaldesk.evals.labeledness.manifest import Protocol
from signaldesk.evals.labeledness.store import (
    AnnotationStore,
    Verdict,
    next_position,
    resolve,
    undo_record,
    verdict_record,
)

pytestmark = pytest.mark.unit

#: Run in a child interpreter that is killed with os._exit, so nothing an
#: orderly shutdown does - atexit, __del__, a context manager, the garbage
#: collector - can be what puts the record on disk. Only the fsync in
#: AnnotationStore.append can.
WRITER_SCRIPT = """
import os
import sys
sys.path.insert(0, {source!r})
from pathlib import Path
from signaldesk.evals.labeledness.manifest import Protocol
from signaldesk.evals.labeledness.store import AnnotationStore, Verdict, verdict_record

AnnotationStore(Path({gold!r})).append(
    verdict_record(
        screen_id="s0001",
        verdict=Verdict.NOT_LABELLED,
        guideline_version="v1",
        protocol=Protocol.FULL,
        elapsed_ms=10,
        set_id="set-1",
        document_id=1,
        section_codes=("adverse_reactions",),
        digests={{"adverse_reactions[0]": "x"}},
    )
)
os._exit(9)
"""


def _verdict(screen_id: str, verdict: Verdict = Verdict.NOT_LABELLED) -> object:
    return verdict_record(
        screen_id=screen_id,
        verdict=verdict,
        guideline_version="v1",
        protocol=Protocol.FULL,
        elapsed_ms=1234,
        set_id=f"set-{screen_id}",
        document_id=1,
        section_codes=("adverse_reactions",),
        digests={"adverse_reactions[0]": "a" * 64},
    )


class TestPersistenceIsPerVerdict:
    def test_a_verdict_survives_a_process_that_never_exits_cleanly(self, tmp_path: Path) -> None:
        """Read back by a fresh interpreter, not by the writer.

        Asserting on the writer's own handle would pass on a buffered write that
        a power cut would lose, which is the failure the fsync is there for. The
        subprocess is killed rather than allowed to exit, so no atexit hook, no
        context-manager teardown and no garbage collector flush can be what makes
        this green.
        """
        gold = tmp_path / "gold.jsonl"
        script = WRITER_SCRIPT.format(source=str(Path("src").resolve()), gold=str(gold))
        completed = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=False
        )
        assert completed.returncode == 9, completed.stderr

        live = AnnotationStore(gold).resolved()
        assert set(live) == {"s0001"}
        assert live["s0001"].verdict is Verdict.NOT_LABELLED

    def test_each_verdict_is_one_line_appended_in_order(self, tmp_path: Path) -> None:
        store = AnnotationStore(tmp_path / "gold.jsonl")
        store.append(_verdict("s0001"))
        store.append(_verdict("s0002", Verdict.LABELLED_EXPLICIT))
        lines = (tmp_path / "gold.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert [json.loads(line)["screen_id"] for line in lines] == ["s0001", "s0002"]


class TestUndoIsATombstone:
    def test_undo_appends_and_never_rewrites(self, tmp_path: Path) -> None:
        """Length strictly increases.

        Pinned on the byte count rather than on the resolved state, so it holds
        independently of how resolution is implemented. If undo ever became a
        seek-and-truncate, the resolved state could still be right while a crash
        mid-undo left a corrupt file.
        """
        path = tmp_path / "gold.jsonl"
        store = AnnotationStore(path)
        store.append(_verdict("s0001"))
        after_verdict = path.stat().st_size
        store.append(undo_record(screen_id="s0001", guideline_version="v1"))
        assert path.stat().st_size > after_verdict

    def test_a_tombstone_removes_the_live_verdict(self, tmp_path: Path) -> None:
        store = AnnotationStore(tmp_path / "gold.jsonl")
        store.append(_verdict("s0001"))
        store.append(undo_record(screen_id="s0001", guideline_version="v1"))
        assert store.resolved() == {}

    def test_a_re_answer_after_an_undo_wins(self, tmp_path: Path) -> None:
        store = AnnotationStore(tmp_path / "gold.jsonl")
        store.append(_verdict("s0001", Verdict.NOT_LABELLED))
        store.append(undo_record(screen_id="s0001", guideline_version="v1"))
        store.append(_verdict("s0001", Verdict.LABELLED_BROADER))
        assert store.resolved()["s0001"].verdict is Verdict.LABELLED_BROADER

    def test_resolution_is_last_write_wins_in_file_order(self) -> None:
        records = [
            _verdict("s0001", Verdict.NOT_LABELLED),
            _verdict("s0002", Verdict.UNCLEAR),
            _verdict("s0001", Verdict.LABELLED_CLASS),
        ]
        live = resolve(records)  # type: ignore[arg-type]
        assert live["s0001"].verdict is Verdict.LABELLED_CLASS
        assert live["s0002"].verdict is Verdict.UNCLEAR


class TestResume:
    def test_it_returns_the_first_screen_with_no_live_verdict(self, tmp_path: Path) -> None:
        store = AnnotationStore(tmp_path / "gold.jsonl")
        store.append(_verdict("s0001"))
        store.append(_verdict("s0002"))
        ids = ["s0001", "s0002", "s0003", "s0004"]
        assert next_position(4, store.resolved(), ids) == 3

    def test_a_screen_retracted_long_ago_is_picked_up_again(self, tmp_path: Path) -> None:
        """Scanning from the front rather than continuing from the high-water mark.

        A resume that continued from the highest answered position would skip a
        retracted screen forever, and the hole would only show up when the eval
        counted the gold set.
        """
        store = AnnotationStore(tmp_path / "gold.jsonl")
        for screen_id in ("s0001", "s0002", "s0003"):
            store.append(_verdict(screen_id))
        store.append(undo_record(screen_id="s0001", guideline_version="v1"))
        assert next_position(3, store.resolved(), ["s0001", "s0002", "s0003"]) == 1

    def test_a_complete_store_resumes_past_the_end(self, tmp_path: Path) -> None:
        store = AnnotationStore(tmp_path / "gold.jsonl")
        store.append(_verdict("s0001"))
        assert next_position(1, store.resolved(), ["s0001"]) == 2


class TestAMalformedStoreStopsTheRun:
    def test_an_unparseable_line_raises_rather_than_being_skipped(self, tmp_path: Path) -> None:
        """A gold set cannot be regenerated, so a bad line is not a line to skip."""
        path = tmp_path / "gold.jsonl"
        path.write_text('{"kind": "verdict", "screen_id": "s1"}\n', encoding="utf-8")
        with pytest.raises(AnnotationError, match="not a valid annotation record"):
            AnnotationStore(path).read_all()

    def test_a_verdict_without_its_provenance_is_rejected_at_construction(self) -> None:
        with pytest.raises(ValueError, match="document and the digests"):
            verdict_record(
                screen_id="s0001",
                verdict=Verdict.NOT_LABELLED,
                guideline_version="v1",
                protocol=Protocol.FULL,
                elapsed_ms=1,
                set_id="set-1",
                document_id=1,
                section_codes=(),
                digests={},
            )


class TestTheBinaryCollapse:
    @pytest.mark.parametrize(
        ("verdict", "expected"),
        [
            (Verdict.LABELLED_EXPLICIT, True),
            (Verdict.LABELLED_BROADER, True),
            (Verdict.LABELLED_CLASS, True),
            (Verdict.NOT_LABELLED, False),
            (Verdict.UNCLEAR, False),
        ],
    )
    def test_the_three_labelled_routes_collapse_together(
        self, verdict: Verdict, expected: bool
    ) -> None:
        assert verdict.is_labelled is expected
