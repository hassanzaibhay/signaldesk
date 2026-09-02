"""The screen-50 report: measured pace, not estimated pace.

Guideline section 8. Two decisions come out of this and both need numbers rather
than impressions: whether the guideline needs a v2 amendment, and what N is
realistic for the day. Every figure here is derived from the ``elapsed_ms`` the
harness recorded per screen, so "how fast am I going" is answered from the log
and not from how it felt.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from signaldesk.evals.labeledness.manifest import Protocol, SampleManifest
from signaldesk.evals.labeledness.store import Record, Verdict


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """What the first N screens measured."""

    annotated: int
    median_seconds: float
    p90_seconds: float
    total_minutes: float
    unclear_rate: float
    verdict_counts: dict[str, int]
    bounded_protocol: int
    bounded_median_seconds: float | None
    full_median_seconds: float | None
    projected_screens_per_hour: float
    #: ``l`` verdicts where the Preferred Term appeared verbatim in the text
    #: shown, and where it did not. None when no manifest was supplied.
    explicit_verbatim: int | None = None
    explicit_by_synonym: int | None = None


def explicit_split(manifest: SampleManifest, records: Sequence[Record]) -> tuple[int, int]:
    """Partition ``l`` verdicts into verbatim-PT and same-entity-different-words.

    This is what the guideline means by there being no sixth verdict for
    synonymy. The distinction between "the label used the Preferred Term" and
    "the label used another name for the same finding" is worth having for error
    analysis, and it costs the annotator nothing, because the manifest holds both
    the PT and the exact text that was rendered. Deriving it afterwards is
    strictly better than asking for it: a keypress would put a fresh judgement
    call in front of the annotator at the point consistency is most fragile.

    Nothing here is shown during annotation. It is computed from the committed
    manifest at scoring time, never in the render path.
    """
    by_id = {screen.screen_id: screen for screen in manifest.screens}
    verbatim = 0
    synonym = 0
    for record in records:
        if record.kind != "verdict" or record.verdict is not Verdict.LABELLED_EXPLICIT:
            continue
        screen = by_id.get(record.screen_id)
        if screen is None:
            continue
        haystack = " ".join(block.text for block in screen.sections).lower()
        if screen.pt.lower() in haystack:
            verbatim += 1
        else:
            synonym += 1
    return verbatim, synonym


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(fraction * len(ordered)))
    return ordered[index]


def summarise(records: Sequence[Record], manifest: SampleManifest | None = None) -> Checkpoint:
    """Reduce the live verdicts to the checkpoint figures.

    ``manifest`` is optional only so the pace figures can be computed without
    one. Pass it whenever it is available: it is what enables the ``l`` split
    described in :func:`explicit_split`.
    """
    verdicts = [record for record in records if record.kind == "verdict"]
    seconds = [(record.elapsed_ms or 0) / 1000.0 for record in verdicts]
    counts = {verdict.value: 0 for verdict in Verdict}
    for record in verdicts:
        if record.verdict is not None:
            counts[record.verdict.value] += 1

    bounded = [
        (record.elapsed_ms or 0) / 1000.0
        for record in verdicts
        if record.protocol is Protocol.BOUNDED
    ]
    full = [
        (record.elapsed_ms or 0) / 1000.0 for record in verdicts if record.protocol is Protocol.FULL
    ]
    median = statistics.median(seconds) if seconds else 0.0
    verbatim, synonym = explicit_split(manifest, verdicts) if manifest else (None, None)
    return Checkpoint(
        annotated=len(verdicts),
        median_seconds=median,
        p90_seconds=_percentile(seconds, 0.9),
        total_minutes=sum(seconds) / 60.0,
        unclear_rate=(counts[Verdict.UNCLEAR.value] / len(verdicts)) if verdicts else 0.0,
        verdict_counts=counts,
        bounded_protocol=len(bounded),
        bounded_median_seconds=statistics.median(bounded) if bounded else None,
        full_median_seconds=statistics.median(full) if full else None,
        projected_screens_per_hour=(3600.0 / median) if median > 0 else 0.0,
        explicit_verbatim=verbatim,
        explicit_by_synonym=synonym,
    )


def render(checkpoint: Checkpoint, manifest: SampleManifest) -> str:
    """The report, formatted for a terminal and phrased as measurements."""
    unclear_flag = "" if checkpoint.unclear_rate < 0.10 else "   ABOVE THE 10 PERCENT TARGET"
    lines = [
        "",
        f"Checkpoint: {checkpoint.annotated} screens annotated under guideline "
        f"{manifest.guideline_version}.",
        "",
        "Pace, from recorded elapsed time and not from an estimate:",
        f"  median            {checkpoint.median_seconds:6.1f} s per screen",
        f"  90th percentile   {checkpoint.p90_seconds:6.1f} s per screen",
        f"  time spent        {checkpoint.total_minutes:6.1f} min",
        f"  implied rate      {checkpoint.projected_screens_per_hour:6.1f} screens per hour",
        "",
        "Verdicts:",
        f"  l labelled, described       {checkpoint.verdict_counts['l']:4}",
        f"  b labelled, broader term    {checkpoint.verdict_counts['b']:4}",
        f"  c labelled, class warning   {checkpoint.verdict_counts['c']:4}",
        f"  n not labelled              {checkpoint.verdict_counts['n']:4}",
        f"  u unclear                   {checkpoint.verdict_counts['u']:4}"
        f"   ({checkpoint.unclear_rate:.0%}){unclear_flag}",
        "",
        "Reading protocol:",
        f"  bounded (long sections)     {checkpoint.bounded_protocol:4} of {checkpoint.annotated}",
    ]
    if checkpoint.bounded_median_seconds is not None:
        lines.append(f"  bounded median              {checkpoint.bounded_median_seconds:6.1f} s")
    if checkpoint.full_median_seconds is not None:
        lines.append(f"  full median                 {checkpoint.full_median_seconds:6.1f} s")
    if checkpoint.explicit_verbatim is not None and checkpoint.verdict_counts["l"]:
        # Derived from the manifest, never asked for. See explicit_split.
        lines += [
            "",
            "Within l, derived after the fact and not by a keypress:",
            f"  the term appeared verbatim  {checkpoint.explicit_verbatim:4}",
            f"  named another way           {checkpoint.explicit_by_synonym:4}",
        ]
    lines += [
        "",
        "Decide two things before continuing:",
        "  1. Does the guideline need a v2 amendment? If yes, amend",
        "     docs/annotation-guideline-labeledness.md with the reason and re-annotate",
        "     these screens under v2.",
        "  2. What N is realistic today, at the measured rate above? The draw is in",
        "     random order, so any prefix is a valid uniform sample and stopping early",
        "     costs representativeness nothing.",
        "",
        "Continue with --past-checkpoint once both are settled.",
        "",
    ]
    return "\n".join(lines)
