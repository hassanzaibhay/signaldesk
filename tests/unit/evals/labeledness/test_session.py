"""The annotation loop: no defaults, per-verdict persistence, the checkpoint."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from _builders import build_manifest, build_screen

from signaldesk.core.errors import AnnotationError
from signaldesk.evals.labeledness.manifest import Protocol, SampleManifest
from signaldesk.evals.labeledness.session import (
    CHECKPOINT_AT,
    MINIMUM_SCORABLE_RECORDS,
    load_session,
    require_scorable,
    run_session,
    verified,
)
from signaldesk.evals.labeledness.store import AnnotationStore, Verdict, undo_record

pytestmark = pytest.mark.unit


class ScriptedKeys:
    """A canned keystroke sequence, so the loop is testable without a terminal."""

    def __init__(self, keys: str, lines: list[str] | None = None) -> None:
        self.keys: Iterator[str] = iter(keys)
        self.lines: Iterator[str] = iter(lines or [])
        self.prompts: list[str] = []

    def read_key(self) -> str:
        return next(self.keys, "q")

    def read_line(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return next(self.lines, "")


def _manifest(count: int = 5) -> SampleManifest:
    return build_manifest(
        [
            build_screen(screen_id=f"s{index:04d}", position=index, pair_id=f"pair-{index}")
            for index in range(1, count + 1)
        ]
    )


class TestNoVerdictIsPreselected:
    def test_bare_enter_writes_nothing_and_re_renders(self, tmp_path: Path) -> None:
        """Asserted on the store, not on the return value.

        A default introduced downstream of the key dispatch -- in the record
        writer, say -- would still leave a record behind while the dispatch
        looked clean. Checking that the file is empty catches that; checking that
        the loop returned no verdict would not.
        """
        store = AnnotationStore(tmp_path / "gold.jsonl")
        frames: list[str] = []
        result = run_session(
            _manifest(),
            store,
            ScriptedKeys("\n\n\nq"),
            write=frames.append,
            past_checkpoint=True,
        )
        assert store.read_all() == []
        assert result.answered == 0
        assert len(frames) == 4

    @pytest.mark.parametrize("key", ["x", "\r", "\t", "0", "L", "Y", "!"])
    def test_an_unrecognised_key_writes_nothing(self, tmp_path: Path, key: str) -> None:
        store = AnnotationStore(tmp_path / "gold.jsonl")
        run_session(
            _manifest(), store, ScriptedKeys(key + "q"), write=lambda _: None, past_checkpoint=True
        )
        assert store.read_all() == []

    def test_uppercase_verdict_keys_are_not_verdicts(self, tmp_path: Path) -> None:
        """Case matters, so a stuck shift key cannot label a screen."""
        store = AnnotationStore(tmp_path / "gold.jsonl")
        run_session(
            _manifest(), store, ScriptedKeys("LBCNUq"), write=lambda _: None, past_checkpoint=True
        )
        assert store.read_all() == []


class TestEachVerdictIsRecordedWithItsProvenance:
    def test_a_verdict_carries_the_document_and_the_rendered_digests(self, tmp_path: Path) -> None:
        manifest = _manifest()
        store = AnnotationStore(tmp_path / "gold.jsonl")
        run_session(manifest, store, ScriptedKeys("nq"), write=lambda _: None, past_checkpoint=True)
        record = store.resolved()["s0001"]
        screen = manifest.screen_at(1)
        assert record.set_id == screen.set_id
        assert record.document_id == screen.document_id
        assert record.section_codes == screen.section_codes
        assert record.digests == screen.digests()
        assert record.guideline_version == "v1"
        assert record.elapsed_ms is not None

    def test_elapsed_time_is_measured_per_screen(self, tmp_path: Path) -> None:
        # start, verdict, start, verdict, then the third screen's start before q.
        ticks = iter([0.0, 2.5, 2.5, 9.0, 9.0])
        store = AnnotationStore(tmp_path / "gold.jsonl")
        run_session(
            _manifest(),
            store,
            ScriptedKeys("nnq"),
            write=lambda _: None,
            past_checkpoint=True,
            clock=lambda: next(ticks),
        )
        live = store.resolved()
        assert live["s0001"].elapsed_ms == 2500
        assert live["s0002"].elapsed_ms == 6500

    def test_a_note_is_prompted_on_unclear_and_only_on_unclear(self, tmp_path: Path) -> None:
        store = AnnotationStore(tmp_path / "gold.jsonl")
        keys = ScriptedKeys("nulq", lines=["the label is ambiguous"])
        run_session(_manifest(), store, keys, write=lambda _: None, past_checkpoint=True)
        live = store.resolved()
        assert live["s0001"].note is None
        assert live["s0002"].note == "the label is ambiguous"
        assert live["s0003"].note is None
        assert len(keys.prompts) == 1

    def test_every_verdict_key_maps_to_its_verdict(self, tmp_path: Path) -> None:
        store = AnnotationStore(tmp_path / "gold.jsonl")
        run_session(
            _manifest(),
            store,
            ScriptedKeys("lbcnu", lines=[""]),
            write=lambda _: None,
            past_checkpoint=True,
        )
        live = store.resolved()
        assert [live[f"s{index:04d}"].verdict for index in range(1, 6)] == [
            Verdict.LABELLED_EXPLICIT,
            Verdict.LABELLED_BROADER,
            Verdict.LABELLED_CLASS,
            Verdict.NOT_LABELLED,
            Verdict.UNCLEAR,
        ]


class TestUndoAndResume:
    def test_undo_steps_back_to_the_screen_it_retracted(self, tmp_path: Path) -> None:
        store = AnnotationStore(tmp_path / "gold.jsonl")
        run_session(
            _manifest(), store, ScriptedKeys("nnzlq"), write=lambda _: None, past_checkpoint=True
        )
        live = store.resolved()
        assert live["s0001"].verdict is Verdict.NOT_LABELLED
        assert live["s0002"].verdict is Verdict.LABELLED_EXPLICIT

    def test_undo_at_the_start_of_a_session_does_nothing(self, tmp_path: Path) -> None:
        store = AnnotationStore(tmp_path / "gold.jsonl")
        run_session(
            _manifest(), store, ScriptedKeys("zzq"), write=lambda _: None, past_checkpoint=True
        )
        assert store.read_all() == []

    def test_a_second_session_resumes_where_the_first_stopped(self, tmp_path: Path) -> None:
        manifest = _manifest()
        path = tmp_path / "gold.jsonl"
        run_session(
            manifest,
            AnnotationStore(path),
            ScriptedKeys("nnq"),
            write=lambda _: None,
            past_checkpoint=True,
        )
        frames: list[str] = []
        run_session(
            manifest,
            AnnotationStore(path),
            ScriptedKeys("nq"),
            write=frames.append,
            past_checkpoint=True,
        )
        assert "screen 3 of 5" in frames[0]
        assert len(AnnotationStore(path).resolved()) == 3

    def test_the_run_reports_when_the_schedule_is_complete(self, tmp_path: Path) -> None:
        result = run_session(
            _manifest(2),
            AnnotationStore(tmp_path / "gold.jsonl"),
            ScriptedKeys("nn"),
            write=lambda _: None,
            past_checkpoint=True,
        )
        assert result.finished is True


class TestTheCheckpoint:
    def test_annotation_stops_at_fifty_screens(self, tmp_path: Path) -> None:
        manifest = _manifest(60)
        store = AnnotationStore(tmp_path / "gold.jsonl")
        result = run_session(manifest, store, ScriptedKeys("n" * 60), write=lambda _: None)
        assert result.reached_checkpoint is True
        assert len(store.resolved()) == CHECKPOINT_AT

    def test_the_flag_carries_past_it(self, tmp_path: Path) -> None:
        manifest = _manifest(60)
        store = AnnotationStore(tmp_path / "gold.jsonl")
        result = run_session(
            manifest, store, ScriptedKeys("n" * 60), write=lambda _: None, past_checkpoint=True
        )
        assert result.reached_checkpoint is False
        assert len(store.resolved()) == 60


class TestTheStoreAndTheManifestMustAgree:
    def test_a_store_from_another_draw_is_refused(self, tmp_path: Path) -> None:
        from signaldesk.evals.labeledness.manifest import write_manifest

        manifest_path = tmp_path / "manifest.json"
        write_manifest(_manifest(), manifest_path)
        gold = tmp_path / "gold.jsonl"
        store = AnnotationStore(gold)
        run_session(
            build_manifest([build_screen(screen_id="zzzz9", position=1)]),
            store,
            ScriptedKeys("n"),
            write=lambda _: None,
            past_checkpoint=True,
        )
        with pytest.raises(AnnotationError, match="different draw"):
            load_session(manifest_path, gold)

    def test_a_verdict_against_text_that_has_moved_is_refused(self, tmp_path: Path) -> None:
        """The check the evaluation loader runs before scoring.

        Scoring a record whose digests disagree with the manifest would attribute
        a judgement to text nobody saw.
        """
        manifest = _manifest()
        store = AnnotationStore(tmp_path / "gold.jsonl")
        run_session(manifest, store, ScriptedKeys("nq"), write=lambda _: None, past_checkpoint=True)
        records = store.read_all()
        assert verified(manifest, records) == records

        moved = build_manifest(
            [
                build_screen(
                    screen_id="s0001",
                    position=1,
                    blocks=[("adverse_reactions", 0, "Different text entirely.")],
                ),
                *manifest.screens[1:],
            ]
        )
        with pytest.raises(AnnotationError, match="different text"):
            verified(moved, records)


class TestNavigationNeverWritesARecord:
    """Every non-verdict key has to be inert with respect to the store.

    This is the same property as "no default verdict" approached from the other
    side: the risk is not only that Enter labels something, it is that a paging
    or search key does.
    """

    def _long_screen(self) -> SampleManifest:
        text = "\n".join(f"line {index} about nausea" for index in range(200))
        return build_manifest(
            [
                build_screen(
                    screen_id="s0001",
                    position=1,
                    blocks=[
                        ("adverse_reactions", 0, text),
                        ("boxed_warning", 0, "A class effect."),
                        ("warnings", 0, "Older format warnings."),
                    ],
                )
            ]
        )

    @pytest.mark.parametrize("keys", [" ", "B", "G", "1", "2", "4", "/", ".", ","])
    def test_a_navigation_key_leaves_the_store_empty(self, tmp_path: Path, keys: str) -> None:
        store = AnnotationStore(tmp_path / "gold.jsonl")
        run_session(
            self._long_screen(),
            store,
            ScriptedKeys(keys + "q", lines=["nausea"]),
            write=lambda _: None,
            past_checkpoint=True,
        )
        assert store.read_all() == []

    def test_paging_moves_the_frame_and_the_top_key_returns_to_it(self, tmp_path: Path) -> None:
        frames: list[str] = []
        run_session(
            self._long_screen(),
            AnnotationStore(tmp_path / "gold.jsonl"),
            ScriptedKeys("  Gq"),
            write=frames.append,
            past_checkpoint=True,
        )
        assert "page 1/" in frames[0]
        assert "page 2/" in frames[1]
        assert "page 3/" in frames[2]
        assert "page 1/" in frames[3]

    def test_a_section_key_switches_only_to_a_section_that_exists(self, tmp_path: Path) -> None:
        frames: list[str] = []
        run_session(
            self._long_screen(),
            AnnotationStore(tmp_path / "gold.jsonl"),
            ScriptedKeys("23q"),
            write=frames.append,
            past_checkpoint=True,
        )
        assert "BOXED WARNING" in frames[1]
        # 3 is warnings_and_cautions, which this document does not carry, so the
        # view must not move rather than falling through to another section.
        assert "BOXED WARNING" in frames[2]

    def test_a_search_reports_its_match_count(self, tmp_path: Path) -> None:
        frames: list[str] = []
        run_session(
            self._long_screen(),
            AnnotationStore(tmp_path / "gold.jsonl"),
            ScriptedKeys("/q", lines=["nausea"]),
            write=frames.append,
            past_checkpoint=True,
        )
        assert 'FIND "nausea"' in frames[1]
        assert "1/200" in frames[1]

    def test_stepping_matches_without_a_query_does_nothing(self, tmp_path: Path) -> None:
        frames: list[str] = []
        run_session(
            self._long_screen(),
            AnnotationStore(tmp_path / "gold.jsonl"),
            ScriptedKeys(".,q"),
            write=frames.append,
            past_checkpoint=True,
        )
        assert frames[0] == frames[1] == frames[2]


class TestNothingIsScoredAgainstNothing:
    """Closes the second limit left open when hygiene rule 3 widened.

    That rule sees untracked files. It cannot see a gold set that was never
    written, or one committed and then emptied, because both present as a clean
    hygiene run. Scoring is where that gets caught instead: an evaluation that
    quietly reports zero against an absent gold set is worse than one that fails,
    because zero looks like a measurement.
    """

    def _answered(self, tmp_path: Path, count: int) -> tuple[SampleManifest, AnnotationStore]:
        manifest = _manifest(max(count, 1))
        store = AnnotationStore(tmp_path / "gold.jsonl")
        if count:
            run_session(
                manifest,
                store,
                ScriptedKeys("n" * count),
                write=lambda _: None,
                past_checkpoint=True,
            )
        return manifest, store

    def test_the_minimum_is_the_checkpoint_and_cannot_drift_from_it(self) -> None:
        assert MINIMUM_SCORABLE_RECORDS == CHECKPOINT_AT

    def test_an_absent_gold_set_fails_rather_than_scoring_zero(self, tmp_path: Path) -> None:
        manifest = _manifest()
        store = AnnotationStore(tmp_path / "never-written.jsonl")
        with pytest.raises(AnnotationError, match="will not score an absent"):
            require_scorable(manifest, store)

    def test_a_gold_set_with_every_verdict_retracted_fails(self, tmp_path: Path) -> None:
        """Records on disk, none of them live. Not the same as an empty file."""
        manifest, store = self._answered(tmp_path, 1)
        store.append(undo_record(screen_id="s0001", guideline_version="v1"))
        assert store.read_all()
        assert store.resolved() == {}
        with pytest.raises(AnnotationError, match="no live verdicts"):
            require_scorable(manifest, store)

    def test_a_gold_set_below_the_minimum_fails_and_says_by_how_much(self, tmp_path: Path) -> None:
        manifest, store = self._answered(tmp_path, 10)
        with pytest.raises(AnnotationError, match="holds 10 live verdicts, fewer than the 50"):
            require_scorable(manifest, store)

    def test_a_gold_set_at_the_minimum_is_scorable(self, tmp_path: Path) -> None:
        manifest, store = self._answered(tmp_path, MINIMUM_SCORABLE_RECORDS)
        assert len(require_scorable(manifest, store)) == MINIMUM_SCORABLE_RECORDS

    def test_the_minimum_is_overridable_for_a_deliberate_partial_score(
        self, tmp_path: Path
    ) -> None:
        """A prefix run stops where the day stops, so the floor has to be movable.

        It is a keyword argument rather than a config value: moving it is a
        decision recorded in the calling code, not a setting that drifts.
        """
        manifest, store = self._answered(tmp_path, 10)
        assert len(require_scorable(manifest, store, minimum=10)) == 10

    def test_it_still_verifies_digests_before_returning(self, tmp_path: Path) -> None:
        """The floor is an addition to the digest check, not a replacement."""
        manifest, store = self._answered(tmp_path, MINIMUM_SCORABLE_RECORDS)
        moved = build_manifest(
            [
                build_screen(
                    screen_id="s0001",
                    position=1,
                    blocks=[("adverse_reactions", 0, "Different text entirely.")],
                ),
                *manifest.screens[1:],
            ]
        )
        with pytest.raises(AnnotationError, match="different text"):
            require_scorable(moved, store)


class TestUndoIsWithinTheSession:
    def test_it_does_not_reach_back_into_a_previous_sitting(self, tmp_path: Path) -> None:
        """Documented behaviour, asserted so it stays documented.

        `z` retracts the verdict just given, which is what a miskey needs. It
        deliberately does not walk backwards through a store written in an
        earlier session: the annotator has no way to see what those verdicts were
        without re-rendering them, so retracting one blind would be worse than
        not offering it. Correcting an old verdict means re-annotating that
        screen, which the resume scan picks up once it is retracted.
        """
        manifest = _manifest()
        path = tmp_path / "gold.jsonl"
        run_session(
            manifest,
            AnnotationStore(path),
            ScriptedKeys("n"),
            write=lambda _: None,
            past_checkpoint=True,
        )
        store = AnnotationStore(path)
        assert len(store.resolved()) == 1

        run_session(manifest, store, ScriptedKeys("zq"), write=lambda _: None, past_checkpoint=True)
        assert len(AnnotationStore(path).resolved()) == 1


class TestTheRecordedGuidelineVersion:
    """A record carries the version its verdict was made under.

    The manifest carries the version the sample was DRAWN under. Conflating them
    was a modelling error, and it made the guideline's own amendment procedure
    unimplementable: the manifest is committed and must not be rewritten to claim
    the sample was drawn against a document that did not exist at the time, so
    without an override no amendment could ever reach a record.
    """

    def test_it_defaults_to_the_manifest(self, tmp_path: Path) -> None:
        store = AnnotationStore(tmp_path / "gold.jsonl")
        run_session(
            _manifest(), store, ScriptedKeys("nq"), write=lambda _: None, past_checkpoint=True
        )
        assert store.resolved()["s0001"].guideline_version == "v1"

    def test_an_override_reaches_every_new_record(self, tmp_path: Path) -> None:
        store = AnnotationStore(tmp_path / "gold.jsonl")
        run_session(
            _manifest(),
            store,
            ScriptedKeys("nnq"),
            write=lambda _: None,
            past_checkpoint=True,
            guideline_version="v2",
        )
        assert {r.guideline_version for r in store.resolved().values()} == {"v2"}

    def test_a_tombstone_carries_the_version_that_retracted_it(self, tmp_path: Path) -> None:
        store = AnnotationStore(tmp_path / "gold.jsonl")
        run_session(
            _manifest(),
            store,
            ScriptedKeys("nzq"),
            write=lambda _: None,
            past_checkpoint=True,
            guideline_version="v2",
        )
        undos = [r for r in store.read_all() if r.kind == "undo"]
        assert undos and all(r.guideline_version == "v2" for r in undos)

    def test_records_written_under_an_older_version_stay_legible_as_that_version(
        self, tmp_path: Path
    ) -> None:
        """The v1 pass is not rewritten by a v2 amendment.

        Append-only means an earlier version stays on disk exactly as written.
        A migration that rewrote old records to the current version would erase
        the fact that they were made under a different rule, which is the one
        thing the version field exists to preserve.
        """
        path = tmp_path / "gold.jsonl"
        run_session(
            _manifest(),
            AnnotationStore(path),
            ScriptedKeys("nn"),
            write=lambda _: None,
            past_checkpoint=True,
        )
        first = AnnotationStore(path).read_all()
        assert {r.guideline_version for r in first} == {"v1"}

        # Retract under v2, the way an amendment mid-project would.
        store = AnnotationStore(path)
        for record in first:
            store.append(undo_record(screen_id=record.screen_id, guideline_version="v2"))

        every = store.read_all()
        assert store.resolved() == {}
        verdicts = [r for r in every if r.kind == "verdict"]
        undos = [r for r in every if r.kind == "undo"]
        assert all(r.guideline_version == "v1" for r in verdicts), "v1 verdicts were rewritten"
        assert all(r.guideline_version == "v2" for r in undos)


class TestElapsedTimeSpansTheWholeScreen:
    """The assertion the entire pace-flag mechanism rests on.

    If ``elapsed_ms`` measured the gap between the last two keypresses rather
    than render-to-verdict, a slowly-paged screen would read fast and a fast one
    slow, every threshold in the checkpoint would become decorative, and nothing
    would look wrong: the numbers would be plausible and simply be about
    something else. It is the only failure route the pace flags cannot see,
    because they consume the very figure that would be corrupted.

    The clock here is driven by the key reader, so time advances on every
    keypress. That is what makes the span observable: under the correct
    implementation the elapsed time covers every navigation key, and under a
    per-keypress reset it would equal one tick.
    """

    class TickingKeys:
        """Keys that advance a clock, so a span can be told from an interval."""

        def __init__(self, keys: str, tick: float, lines: list[str] | None = None) -> None:
            self._keys: Iterator[str] = iter(keys)
            self._lines: Iterator[str] = iter(lines or [])
            self.tick = tick
            self.now = 0.0

        def read_key(self) -> str:
            self.now += self.tick
            return next(self._keys, "q")

        def read_line(self, prompt: str) -> str:
            return next(self._lines, "")

        def clock(self) -> float:
            return self.now

    def _run(self, tmp_path: Path, manifest: SampleManifest) -> AnnotationStore:
        store = AnnotationStore(tmp_path / "gold.jsonl")
        # page down, section switch, search, then the verdict: four keys, and the
        # elapsed time must cover all four rather than the last one.
        keys = self.TickingKeys(" 2/n", tick=10.0, lines=["nausea"])
        run_session(
            manifest,
            store,
            keys,
            write=lambda _: None,
            past_checkpoint=True,
            clock=keys.clock,
        )
        return store

    def test_a_full_screen_records_render_to_verdict(self, tmp_path: Path) -> None:
        manifest = build_manifest(
            [
                build_screen(
                    screen_id="s0001",
                    position=1,
                    blocks=[
                        ("adverse_reactions", 0, "Nausea was reported. " * 20),
                        ("boxed_warning", 0, "A class effect."),
                    ],
                )
            ]
        )
        record = self._run(tmp_path, manifest).resolved()["s0001"]
        assert record.protocol is Protocol.FULL
        # Four keypresses at 10 s each, measured from the render before the first.
        assert record.elapsed_ms == 40_000
        assert record.elapsed_ms != 10_000, "elapsed covers only the last interval"

    def test_a_bounded_screen_records_render_to_verdict(self, tmp_path: Path) -> None:
        manifest = build_manifest(
            [
                build_screen(
                    screen_id="s0001",
                    position=1,
                    blocks=[
                        ("adverse_reactions", 0, "Nausea and other findings. " * 800),
                        ("boxed_warning", 0, "A class effect."),
                    ],
                )
            ]
        )
        record = self._run(tmp_path, manifest).resolved()["s0001"]
        assert record.protocol is Protocol.BOUNDED
        assert record.elapsed_ms == 40_000
        assert record.elapsed_ms != 10_000, "elapsed covers only the last interval"

    def test_navigation_alone_does_not_shorten_the_span(self, tmp_path: Path) -> None:
        """More paging must mean more elapsed time, not the same.

        Under a per-keypress reset both of these read one tick and the assertion
        below is what separates them.
        """
        manifest = build_manifest(
            [
                build_screen(
                    screen_id="s0001", position=1, blocks=[("adverse_reactions", 0, "x " * 200)]
                )
            ]
        )
        quick = AnnotationStore(tmp_path / "quick.jsonl")
        keys = self.TickingKeys("n", tick=10.0)
        run_session(
            manifest, quick, keys, write=lambda _: None, past_checkpoint=True, clock=keys.clock
        )
        slow = AnnotationStore(tmp_path / "slow.jsonl")
        keys = self.TickingKeys("      n", tick=10.0)
        run_session(
            manifest, slow, keys, write=lambda _: None, past_checkpoint=True, clock=keys.clock
        )

        assert quick.resolved()["s0001"].elapsed_ms == 10_000
        assert slow.resolved()["s0001"].elapsed_ms == 70_000
