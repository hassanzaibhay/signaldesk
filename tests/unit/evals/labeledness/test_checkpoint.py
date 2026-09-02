"""The screen-50 report, computed from recorded time rather than estimated."""

from __future__ import annotations

import pytest
from _builders import build_manifest, build_screen

from signaldesk.evals.labeledness.checkpoint import explicit_split, render, summarise
from signaldesk.evals.labeledness.manifest import Protocol, SampleManifest
from signaldesk.evals.labeledness.store import Record, Verdict, verdict_record

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
