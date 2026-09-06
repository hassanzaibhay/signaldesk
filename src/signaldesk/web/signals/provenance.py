"""The committed artifacts behind the numbers the page prints.

Every published figure on this page is read out of a file under
``evals/history/`` rather than recomputed here, so the page cannot drift from
what the repository claims. The run's own row-level statistics come from its
Parquet, and the banner names the run id, the commit the build ran at, and the
artifact filename, which is what makes any one of those rows traceable.

Two rules from the artifacts themselves are enforced in code rather than left to
a template author:

* MGPS is withheld. ``quotable.withheld`` lists it and ``quotable.estimators``
  does not, so ``withheld_reason`` is carried through verbatim and there is no
  accessor on this module that returns an EBGM or EBGM05 number.
* The two SPL hit rates are quotable together and never one alone - the artifact
  says so in its own ``rate_note``. There is therefore no property that returns
  one of them formatted for display; ``rates_sentence`` returns both in one
  string or nothing at all.

The count of label-carrying drug strings is a count of strings. The same
artifact retires 191/200 as a double-counted 95.5 percent, so nothing here
divides it by anything.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from signaldesk.analytics.signals import history_root
from signaldesk.core.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RunProvenance:
    """What the committed signal artifact says about the run being served."""

    artifact: str
    run_id: str
    code_sha: str
    created_at: str
    min_a: int
    pairs_written: int
    pairs_at_or_above_min_a: int
    flagged_pairs: int
    quotable_estimators: tuple[str, ...]
    withheld_estimators: tuple[str, ...]
    withheld_reason: str

    @property
    def short_sha(self) -> str:
        return self.code_sha[:12]


@dataclass(frozen=True, slots=True)
class LabelProvenance:
    """What the committed SPL artifact says about label coverage."""

    artifact: str
    run_id: str
    selected_slots: int
    distinct_queries_selected: int
    distinct_queries_resolved: int
    hit_rate_distinct_queries: float
    hit_rate_selected_slots: float
    #: Drug strings that reached at least one label. A count, never a rate.
    label_carrying_strings: int
    total_flagged_strings: int
    distinct_documents: int
    distinct_sections: int

    @property
    def rates_sentence(self) -> str:
        """Both hit rates, in one sentence, or nothing.

        The artifact's own ``rate_note`` says the two rates "are measured and
        quotable, together and never one alone", because they share a numerator
        and answer different questions: one asks whether the query strategy
        works, the other asks what the 200-slot cap actually delivered. Returning
        them as a single string is how that rule is kept mechanically instead of
        by asking every template to remember it.
        """
        return (
            f"{self.distinct_queries_resolved} of {self.distinct_queries_selected} "
            f"distinct queries resolved ({self.hit_rate_distinct_queries * 100:.1f} percent), "
            f"which is {self.distinct_queries_resolved} of the {self.selected_slots} "
            f"selected slots ({self.hit_rate_selected_slots * 100:.1f} percent)"
        )


def _load(path: Path) -> dict[str, Any]:
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return document


def _newest(prefix: str) -> Path | None:
    """The most recent artifact with this prefix.

    Filenames carry a UTC timestamp in a sortable format, so the newest is the
    last by name. Returns None rather than raising: a machine that has not run
    an ingest has no artifact, and the page says so instead of failing.
    """
    paths = sorted(history_root().glob(f"{prefix}_*.json"))
    return paths[-1] if paths else None


@lru_cache(maxsize=8)
def run_provenance(run_id: str) -> RunProvenance | None:
    """The committed record for one run, or None if no artifact names it.

    Searched by run id across every signals artifact rather than assuming the
    newest one holds it: artifacts accumulate and a run stays in whichever one
    recorded it. None is a real answer - a run built locally and not yet
    collected into an artifact has no published numbers, and the banner says
    exactly that rather than showing figures that trace nowhere.
    """
    for path in sorted(history_root().glob("signals_*.json"), reverse=True):
        document = _load(path)
        runs = document.get("runs")
        if not isinstance(runs, list):
            continue
        for run in runs:
            if not isinstance(run, dict) or run.get("run_id") != run_id:
                continue
            return _run_provenance(path, run, document)
    log.warning("signals.provenance.absent", run_id=run_id, root=str(history_root()))
    return None


def _run_provenance(path: Path, run: dict[str, Any], document: dict[str, Any]) -> RunProvenance:
    pairs = run.get("pairs", {})
    params = run.get("params", {})
    counts = run.get("flag_counts", {})
    quotable = document.get("quotable", {})
    mgps = run.get("mgps", {})
    return RunProvenance(
        artifact=path.name,
        run_id=str(run["run_id"]),
        code_sha=str(run.get("code_sha", "unknown")),
        created_at=str(run.get("created_at", "")),
        min_a=int(params.get("min_a", 0)),
        pairs_written=int(pairs.get("observed_and_written", 0)),
        pairs_at_or_above_min_a=int(pairs.get("at_or_above_min_a", 0)),
        flagged_pairs=int(counts.get("ror_prr_bcpnn") or 0),
        quotable_estimators=tuple(quotable.get("estimators", ())),
        withheld_estimators=tuple(quotable.get("withheld", ())),
        # The reason is carried verbatim. Paraphrasing the artifact's own
        # account of why a number is not a measurement is how a caveat turns
        # into a footnote and then into nothing.
        withheld_reason=str(quotable.get("withheld_reason") or mgps.get("note") or ""),
    )


@lru_cache(maxsize=1)
def label_provenance() -> LabelProvenance | None:
    """The committed SPL ingest record, or None if there is no artifact."""
    path = _newest("spl_ingest")
    if path is None:
        log.warning("signals.provenance.labels.absent", root=str(history_root()))
        return None
    document = _load(path)
    resolution = document.get("resolution", {})
    scope = document.get("scope", {})
    stored = document.get("stored", {})
    return LabelProvenance(
        artifact=path.name,
        run_id=str(document.get("run_id", "")),
        selected_slots=int(scope.get("selected", 0)),
        distinct_queries_selected=int(resolution.get("distinct_queries_selected", 0)),
        distinct_queries_resolved=int(resolution.get("distinct_queries_resolved", 0)),
        hit_rate_distinct_queries=float(resolution.get("hit_rate_distinct_queries", 0.0)),
        hit_rate_selected_slots=float(resolution.get("hit_rate_selected_slots", 0.0)),
        label_carrying_strings=int(resolution.get("hit_at_least_one_label", 0)),
        total_flagged_strings=int(scope.get("n_total_flagged_strings", 0)),
        distinct_documents=int(stored.get("distinct_documents", 0)),
        distinct_sections=int(stored.get("distinct_sections", 0)),
    )
