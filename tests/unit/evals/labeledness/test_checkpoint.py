"""The screen-50 report, computed from recorded time rather than estimated."""

from __future__ import annotations

from pathlib import Path

import pytest
from _builders import build_manifest, build_screen

from signaldesk.evals.labeledness.checkpoint import (
    IMPLAUSIBLE_SHARE,
    SLOW_MEDIAN_SECONDS,
    explicit_split,
    render,
    summarise,
)
from signaldesk.evals.labeledness.manifest import Protocol, SampleManifest
from signaldesk.evals.labeledness.store import (
    AnnotationStore,
    Record,
    Verdict,
    undo_record,
    verdict_record,
)

pytestmark = pytest.mark.unit


def _record(
    index: int,
    verdict: Verdict,
    seconds: float,
    protocol: Protocol = Protocol.FULL,
) -> Record:
    return verdict_record(
        screen_id=f"s{index:04d}",
        verdict=verdict,
        guideline_version="v1",
        protocol=protocol,
        elapsed_ms=int(seconds * 1000),
        set_id=f"set-{index}",
        document_id=index,
        section_codes=("adverse_reactions",),
        digests={"adverse_reactions[0]": "x" * 64},
    )


class TestPaceComesFromTheLog:
    def test_the_median_and_p90_are_computed_from_elapsed_ms(self) -> None:
        records = [
            _record(index, Verdict.NOT_LABELLED, seconds)
            for index, seconds in enumerate([30, 60, 90, 120, 600], start=1)
        ]
        checkpoint = summarise(records)
        assert checkpoint.annotated == 5
        assert checkpoint.median_seconds == 90.0
        assert checkpoint.p90_seconds == 600.0
        assert checkpoint.total_minutes == pytest.approx(15.0)

    def test_the_implied_rate_follows_the_median(self) -> None:
        records = [_record(index, Verdict.NOT_LABELLED, 90.0) for index in range(1, 4)]
        assert summarise(records).projected_screens_per_hour == pytest.approx(40.0)

    def test_an_empty_store_reports_zero_rather_than_dividing(self) -> None:
        checkpoint = summarise([])
        assert checkpoint.annotated == 0
        assert checkpoint.projected_screens_per_hour == 0.0
        assert checkpoint.unclear_rate == 0.0


class TestTheVerdictMix:
    def test_the_route_split_and_unclear_rate_are_reported(self) -> None:
        records = [
            _record(1, Verdict.LABELLED_EXPLICIT, 40),
            _record(2, Verdict.LABELLED_BROADER, 50),
            _record(3, Verdict.LABELLED_CLASS, 60),
            _record(4, Verdict.NOT_LABELLED, 70),
            _record(5, Verdict.UNCLEAR, 80),
        ]
        checkpoint = summarise(records)
        assert checkpoint.verdict_counts == {"l": 1, "b": 1, "c": 1, "n": 1, "u": 1}
        assert checkpoint.unclear_rate == pytest.approx(0.2)

    def test_an_unclear_rate_over_the_target_is_flagged_in_the_report(self) -> None:
        records = [_record(index, Verdict.UNCLEAR, 60) for index in range(1, 4)]
        text = render(summarise(records), build_manifest([build_screen()]))
        assert "ABOVE THE 10 PERCENT TARGET" in text

    def test_a_rate_under_the_target_is_not_flagged(self) -> None:
        records = [_record(index, Verdict.NOT_LABELLED, 60) for index in range(1, 21)]
        text = render(summarise(records), build_manifest([build_screen()]))
        assert "ABOVE THE 10 PERCENT TARGET" not in text


class TestTheProtocolSplit:
    def test_long_and_short_sections_are_timed_separately(self) -> None:
        """So the eval can say whether bounded verdicts behave differently."""
        records = [
            _record(1, Verdict.NOT_LABELLED, 60, Protocol.FULL),
            _record(2, Verdict.NOT_LABELLED, 80, Protocol.FULL),
            _record(3, Verdict.NOT_LABELLED, 150, Protocol.BOUNDED),
        ]
        checkpoint = summarise(records)
        assert checkpoint.bounded_protocol == 1
        assert checkpoint.bounded_median_seconds == 150.0
        assert checkpoint.full_median_seconds == 70.0

    def test_a_report_without_bounded_screens_omits_that_line(self) -> None:
        records = [_record(1, Verdict.NOT_LABELLED, 60, Protocol.FULL)]
        text = render(summarise(records), build_manifest([build_screen()]))
        assert "bounded median" not in text
        assert "full median" in text


class TestTheReportSaysWhatToDecide:
    def test_it_names_both_decisions_and_the_guideline_file(self) -> None:
        text = render(summarise([]), build_manifest([build_screen()]))
        assert "docs/annotation-guideline-labeledness.md" in text
        assert "--past-checkpoint" in text
        assert "any prefix is a valid uniform sample" in text


class TestTheExplicitSplitIsDerivedNotAsked:
    """No sixth verdict for synonymy; the distinction is recovered afterwards.

    The manifest holds the Preferred Term and the exact rendered text, so
    partitioning `l` into "the label used the PT" and "the label used another
    name for the same finding" costs the annotator nothing. Asking for it would
    put a fresh judgement call in front of them at the point consistency is most
    fragile.
    """

    def _manifest_and_records(self) -> tuple[SampleManifest, list[Record]]:
        manifest = build_manifest(
            [
                build_screen(
                    screen_id="s0001",
                    position=1,
                    pt="URTICARIA",
                    blocks=[("adverse_reactions", 0, "Urticaria was reported.")],
                ),
                build_screen(
                    screen_id="s0002",
                    position=2,
                    pt="URTICARIA",
                    blocks=[("adverse_reactions", 0, "Hives were reported.")],
                ),
                build_screen(
                    screen_id="s0003",
                    position=3,
                    pt="MYOCARDIAL INFARCTION",
                    blocks=[("adverse_reactions", 0, "Heart attack has occurred.")],
                ),
                build_screen(
                    screen_id="s0004",
                    position=4,
                    pt="HEADACHE",
                    blocks=[("adverse_reactions", 0, "Nothing relevant here.")],
                ),
            ]
        )
        records = [
            _record(1, Verdict.LABELLED_EXPLICIT, 30),
            _record(2, Verdict.LABELLED_EXPLICIT, 40),
            _record(3, Verdict.LABELLED_EXPLICIT, 50),
            _record(4, Verdict.NOT_LABELLED, 60),
        ]
        return manifest, records

    def test_verbatim_and_synonym_l_verdicts_are_told_apart(self) -> None:
        manifest, records = self._manifest_and_records()
        verbatim, synonym = explicit_split(manifest, records)
        assert verbatim == 1
        assert synonym == 2

    def test_only_l_verdicts_are_partitioned(self) -> None:
        """b and c are labelled too, and neither is what this split is about."""
        manifest = build_manifest(
            [
                build_screen(
                    screen_id="s0001",
                    position=1,
                    pt="HEADACHE",
                    blocks=[("adverse_reactions", 0, "Headache.")],
                )
            ]
        )
        for verdict in (Verdict.LABELLED_BROADER, Verdict.LABELLED_CLASS, Verdict.NOT_LABELLED):
            assert explicit_split(manifest, [_record(1, verdict, 30)]) == (0, 0)

    def test_the_split_is_reported_only_when_a_manifest_is_given(self) -> None:
        manifest, records = self._manifest_and_records()
        assert summarise(records).explicit_verbatim is None
        assert summarise(records, manifest).explicit_verbatim == 1

    def test_the_report_shows_the_split_and_says_it_was_not_a_keypress(self) -> None:
        manifest, records = self._manifest_and_records()
        text = render(summarise(records, manifest), manifest)
        assert "the term appeared verbatim" in text
        assert "named another way" in text
        assert "not by a keypress" in text

    def test_a_report_with_no_l_verdicts_omits_the_split(self) -> None:
        manifest, _records = self._manifest_and_records()
        text = render(summarise([_record(4, Verdict.NOT_LABELLED, 60)], manifest), manifest)
        assert "the term appeared verbatim" not in text

    def test_matching_is_case_insensitive_and_spans_every_section(self) -> None:
        manifest = build_manifest(
            [
                build_screen(
                    screen_id="s0001",
                    position=1,
                    pt="MYOCARDIAL INFARCTION",
                    blocks=[
                        ("adverse_reactions", 0, "Nothing here."),
                        ("boxed_warning", 0, "Risk of myocardial infarction."),
                    ],
                )
            ]
        )
        assert explicit_split(manifest, [_record(1, Verdict.LABELLED_EXPLICIT, 30)]) == (1, 0)


class TestThePaceFlag:
    """Flags, not blocks.

    A hard floor buys better-looking telemetry and the same defect, because
    waiting out a timer is easier than reading a label. A line saying how many
    screens were implausibly fast is harder to ignore and cannot be satisfied
    without actually slowing down.

    The thresholds exist because a real pass was discarded that the report did
    not flag: 50 screens at a 0.20 s median, 49 of them under 5 s, and the only
    conditional line in the report keyed on the unclear rate, which was zero.
    Everything printed was correct and nothing said the pass was impossible.
    """

    def test_it_fires_on_the_pass_that_prompted_it(self) -> None:
        """The discarded pass, reconstructed from its measured figures."""
        records = [_record(index, Verdict.NOT_LABELLED, 0.20) for index in range(1, 50)]
        records.append(_record(50, Verdict.NOT_LABELLED, 299.0))
        checkpoint = summarise(records)

        assert checkpoint.implausible_screens == 49
        assert checkpoint.implausible_share == pytest.approx(0.98)
        assert checkpoint.pace_is_implausible is True

        text = render(checkpoint, build_manifest([build_screen()]))
        assert "PACE IS IMPLAUSIBLE FOR THE READING PROTOCOL" in text
        assert "49 of 50 screens" in text
        assert "flag, not a block" in text

    def test_a_plausible_pass_stays_quiet(self) -> None:
        records = [_record(index, Verdict.NOT_LABELLED, 75.0) for index in range(1, 21)]
        checkpoint = summarise(records)
        assert checkpoint.implausible_screens == 0
        assert checkpoint.pace_is_implausible is False
        assert "PACE IS IMPLAUSIBLE" not in render(checkpoint, build_manifest([build_screen()]))

    def test_the_count_is_reported_even_when_the_flag_does_not_fire(self) -> None:
        """How many, not merely whether. A count survives being argued with."""
        records = [_record(index, Verdict.NOT_LABELLED, 75.0) for index in range(1, 20)]
        records.append(_record(20, Verdict.NOT_LABELLED, 2.0))
        checkpoint = summarise(records)
        assert checkpoint.implausible_screens == 1
        assert checkpoint.pace_is_implausible is False
        assert "1 of 20 screens (5%)" in render(checkpoint, build_manifest([build_screen()]))

    def test_a_slow_median_alone_fires_it(self) -> None:
        """Either threshold is sufficient; they catch different shapes."""
        records = [_record(index, Verdict.NOT_LABELLED, 12.0) for index in range(1, 21)]
        checkpoint = summarise(records)
        assert checkpoint.implausible_screens == 0
        assert checkpoint.pace_is_implausible is True

    def test_a_fast_minority_alone_fires_it(self) -> None:
        records = [_record(index, Verdict.NOT_LABELLED, 90.0) for index in range(1, 18)]
        records += [_record(index, Verdict.NOT_LABELLED, 1.0) for index in range(18, 21)]
        checkpoint = summarise(records)
        assert checkpoint.median_seconds >= SLOW_MEDIAN_SECONDS
        assert checkpoint.implausible_share > IMPLAUSIBLE_SHARE
        assert checkpoint.pace_is_implausible is True

    def test_an_empty_store_does_not_fire(self) -> None:
        """Zero screens is not a fast pass, and 0/0 must not divide."""
        checkpoint = summarise([])
        assert checkpoint.pace_is_implausible is False
        assert checkpoint.implausible_share == 0.0

    @pytest.mark.parametrize(
        ("seconds", "counted"),
        [(4.999, True), (5.0, False), (5.001, False)],
    )
    def test_the_per_screen_boundary_is_strictly_under(self, seconds: float, counted: bool) -> None:
        checkpoint = summarise([_record(1, Verdict.NOT_LABELLED, seconds)])
        assert (checkpoint.implausible_screens == 1) is counted


class TestTheRecordedGuidelineVersionIsReported:
    """The report must name the version the verdicts were MADE under.

    Reporting the manifest's version would misattribute every record after an
    amendment: the manifest carries the version the sample was DRAWN under, and
    those diverge the moment the guideline is amended.
    """

    def test_the_version_comes_from_the_records_not_the_manifest(self) -> None:
        v2 = [
            verdict_record(
                screen_id="s0001",
                verdict=Verdict.NOT_LABELLED,
                guideline_version="v2",
                protocol=Protocol.FULL,
                elapsed_ms=60_000,
                set_id="set-1",
                document_id=1,
                section_codes=("adverse_reactions",),
                digests={"adverse_reactions[0]": "x" * 64},
            )
        ]
        manifest = build_manifest([build_screen()])  # drawn under v1
        text = render(summarise(v2, manifest), manifest)
        assert "annotated under guideline v2" in text
        assert "sample drawn under v1" in text

    def test_records_spanning_two_versions_are_flagged_not_pooled(self) -> None:
        mixed = [
            _record(1, Verdict.NOT_LABELLED, 60),
            verdict_record(
                screen_id="s0002",
                verdict=Verdict.NOT_LABELLED,
                guideline_version="v2",
                protocol=Protocol.FULL,
                elapsed_ms=60_000,
                set_id="set-2",
                document_id=2,
                section_codes=("adverse_reactions",),
                digests={"adverse_reactions[0]": "y" * 64},
            ),
        ]
        checkpoint = summarise(mixed)
        assert checkpoint.guideline_versions == ("v1", "v2")
        text = render(checkpoint, build_manifest([build_screen()]))
        assert "RECORDS SPAN MORE THAN ONE GUIDELINE VERSION" in text
        assert "not pooled" in text


class TestSummariseSeesLiveRecordsOnly:
    """Closes the route by which the pace flag reads quiet on a store that should trip.

    `summarise` filters to verdict records but has no view of retraction: an undo
    is a separate record, and the fast verdict it retracts is still on disk.
    Whether the flag fires therefore depends on the caller handing over the live
    set rather than the whole log, and nothing asserted that until now.

    The two assertions below are deliberately opposite. Passing the live set must
    stay quiet; passing the raw log must trip. If both were quiet, the
    distinction would not exist and the test would prove nothing.
    """

    def _store(self, tmp_path: Path) -> AnnotationStore:
        store = AnnotationStore(tmp_path / "gold.jsonl")
        # Ten fast verdicts, all retracted - the discarded pass in miniature.
        for index in range(1, 11):
            store.append(_record(index, Verdict.NOT_LABELLED, 0.2))
            store.append(undo_record(screen_id=f"s{index:04d}", guideline_version="v1"))
        # Twenty plausible ones that stand.
        for index in range(11, 31):
            store.append(_record(index, Verdict.NOT_LABELLED, 80.0))
        return store

    def test_the_live_set_excludes_retracted_fast_verdicts(self, tmp_path: Path) -> None:
        live = list(self._store(tmp_path).resolved().values())
        checkpoint = summarise(live)

        assert checkpoint.annotated == 20
        assert checkpoint.median_seconds == 80.0
        assert checkpoint.implausible_screens == 0
        assert checkpoint.implausible_share == 0.0
        assert checkpoint.pace_is_implausible is False

    def test_the_raw_log_would_trip_the_flag(self, tmp_path: Path) -> None:
        """The other half. The caller's choice is load-bearing, so it is shown."""
        every = self._store(tmp_path).read_all()
        checkpoint = summarise(every)

        assert checkpoint.annotated == 30
        assert checkpoint.implausible_screens == 10
        assert checkpoint.implausible_share > IMPLAUSIBLE_SHARE
        assert checkpoint.pace_is_implausible is True

    def test_a_retracted_then_re_answered_screen_counts_once_at_its_new_pace(
        self, tmp_path: Path
    ) -> None:
        """Last write wins, so a corrected verdict is timed by the correction."""
        store = AnnotationStore(tmp_path / "gold.jsonl")
        store.append(_record(1, Verdict.NOT_LABELLED, 0.2))
        store.append(undo_record(screen_id="s0001", guideline_version="v1"))
        store.append(_record(1, Verdict.LABELLED_EXPLICIT, 95.0))

        checkpoint = summarise(list(store.resolved().values()))
        assert checkpoint.annotated == 1
        assert checkpoint.median_seconds == 95.0
        assert checkpoint.implausible_screens == 0
