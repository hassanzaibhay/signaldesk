"""The signal build: contingency table to scored, provenance-carrying rows.

One run produces one immutable set of rows under a ``run_id``, plus the record
of what produced them. Nothing overwrites a previous run, so a number quoted
anywhere can be traced to the exact parameters, corpus and commit behind it.
The run record carries:

* the analysis parameters, from ``ContingencySpec``;
* the commit the code was at;
* the data snapshot - which quarters, at which ingest checksums;
* the fitted MGPS hyperparameters, because the shrinkage is not reproducible
  without them;
* wall clock and peak resident memory, measured rather than estimated.

Output is Parquet under ``DATA_DIR``, not Postgres. The web layer that will
serve these rows does not exist yet, and a Django model plus migration created
now would collide with the tracks that own those files. See ADR 0004 for why the
high-cardinality analytical tables live in Parquet regardless.
"""

from __future__ import annotations

import json
import os
import platform
import resource
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from signaldesk.analytics.contingency import ContingencySpec, PairTable
from signaldesk.analytics.contingency import build as build_contingency
from signaldesk.core.config import Settings, get_settings
from signaldesk.core.logging import get_logger
from signaldesk.stats.panel import estimate_all
from signaldesk.stats.types import EstimatorPanel

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SignalRun:
    """What one build did, and everything needed to reproduce it."""

    run_id: str
    created_at: str
    params: dict[str, str | int | None]
    code_sha: str
    data_snapshot: dict[str, str]
    n_cases: int
    pairs_observed: int
    pairs_sufficient: int
    duplicates_removed: int
    cases_without_drug: int
    cases_without_reaction: int
    hyperparameters: dict[str, object]
    mgps_provisional: bool
    flag_counts: dict[str, int | None]
    ic_divergence: dict[str, float]
    seconds: float
    peak_rss_bytes: int

    def as_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "params": self.params,
            "code_sha": self.code_sha,
            "data_snapshot": self.data_snapshot,
            "population": {
                "deduplicated_cases": self.n_cases,
                "duplicates_removed": self.duplicates_removed,
                "cases_with_no_drug_row_in_scope": self.cases_without_drug,
                "cases_with_no_coded_term": self.cases_without_reaction,
            },
            "pairs": {
                "observed_and_written": self.pairs_observed,
                "at_or_above_min_a": self.pairs_sufficient,
            },
            "mgps": {
                "hyperparameters": self.hyperparameters,
                "provisional": self.mgps_provisional,
                "note": (
                    "The gamma mixture prior converged onto a bound, so EBGM and "
                    "EBGM05 are written but are not measurements and must not be "
                    "quoted. flag_all_four is null for the same reason; "
                    "flag_three_of_four is what this run supports."
                )
                if self.mgps_provisional
                else "Interior optimum; the prior is an estimate.",
            },
            "flag_counts": self.flag_counts,
            "ic_divergence": self.ic_divergence,
            "performance": {
                "seconds": round(self.seconds, 2),
                "peak_rss_bytes": self.peak_rss_bytes,
                "peak_rss_gib": round(self.peak_rss_bytes / 1024**3, 3),
                "platform": platform.platform(),
            },
        }


def signal_root(settings: Settings | None = None) -> Path:
    """Where scored pairs are written, partitioned by run."""
    settings = settings or get_settings()
    return settings.data_dir / "parquet" / "signal"


def run_root(settings: Settings | None = None) -> Path:
    """Where the run records are written."""
    settings = settings or get_settings()
    return settings.data_dir / "parquet" / "signal_run"


def _code_sha() -> str:
    """The commit the build ran at, or a marker saying it could not be read.

    ``.git`` is deliberately not mounted into the container - the application
    has no business reading the working tree - so ``git rev-parse`` cannot work
    there and the Makefile passes the host's answer in as
    ``SIGNALDESK_CODE_SHA``. The subprocess path is the fallback for a run
    invoked outside the container.

    An empty string here would look like a value. ``unknown`` says the
    provenance is incomplete, which is a different and worse thing than a run
    that has not been committed yet.
    """
    from_environment = os.environ.get("SIGNALDESK_CODE_SHA", "").strip()
    if from_environment:
        return from_environment

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _data_snapshot() -> dict[str, str]:
    """Quarter to ingest checksum, for every completed FAERS unit.

    This is what makes a run reproducible: the same quarters at the same
    upstream checksums give the same table. A rerun after an upstream revision
    is a different run and must be visibly so.
    """
    from signaldesk.web.signals.models import IngestManifest

    return {
        row.unit: row.checksum
        for row in IngestManifest.objects.filter(
            source="faers", status=IngestManifest.Status.COMPLETED
        ).order_by("unit")
    }


def _peak_rss_bytes() -> int:
    """Peak resident set size of this process.

    ``ru_maxrss`` is kilobytes on Linux and bytes on macOS. The container is
    Linux, and the multiplier is stated rather than guessed at read time.
    """
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return int(usage.ru_maxrss) * (1 if platform.system() == "Darwin" else 1024)


def _rows(pairs: PairTable, panel: EstimatorPanel, run_id: str) -> pl.DataFrame:
    table = panel.contingency
    return pl.DataFrame(
        {
            "run_id": [run_id] * len(pairs),
            "drug": pairs.drug,
            "pt": pairs.pt,
            "a": table.a,
            "b": table.b,
            "c": table.c,
            "d": table.d,
            "n_cases": table.n,
            "expected": table.expected,
            "ror": panel.ror.ror,
            "ror_lower": panel.ror.ror_lower,
            "ror_upper": panel.ror.ror_upper,
            "prr": panel.prr.prr,
            "prr_lower": panel.prr.prr_lower,
            "prr_upper": panel.prr.prr_upper,
            "chi2": panel.prr.chi2,
            "ic": panel.bcpnn.ic,
            "ic025": panel.bcpnn.ic025,
            "ic_observed_expected": panel.bcpnn.ic_observed_expected,
            "ic025_exact": panel.bcpnn.ic025_exact,
            "ebgm": panel.mgps.ebgm,
            "ebgm05": panel.mgps.ebgm05,
            "qn": panel.mgps.qn,
            "corrected": panel.ror.corrected,
            "insufficient": panel.insufficient,
            "flag_ror": panel.flag_ror,
            "flag_prr": panel.flag_prr,
            "flag_bcpnn": panel.flag_bcpnn,
            "flag_mgps": panel.flag_mgps,
            "flag_three_of_four": panel.flag_three_of_four,
            "flag_all_four": (
                panel.flag_all_four if panel.flag_all_four is not None else [None] * len(pairs)
            ),
        }
    )


def _flag_counts(panel: EstimatorPanel) -> dict[str, int | None]:
    """Flag counts, headline and total.

    ARCHITECTURE section 8.1 excludes pairs below the minimum cell count from
    headline counts. Only the ROR rule carries that minimum in its own
    definition; PRR, BCPNN and MGPS do not, so on the full corpus PRR alone
    flags more pairs than there are pairs with three cases. Reporting that
    number as a headline would be reporting a count of signals that the same
    document says are not signals.

    So both are recorded. Keys without a suffix are the headline counts, over
    pairs at or above the minimum. The ``_including_insufficient`` keys are the
    raw threshold counts, kept because the gap between the two is itself worth
    seeing.
    """
    sufficient = ~panel.insufficient

    counts: dict[str, int | None] = {
        "insufficient": int(np.sum(panel.insufficient)),
        "haldane_anscombe_corrected": int(np.sum(panel.ror.corrected)),
    }
    for name, flag in (
        ("ror", panel.flag_ror),
        ("prr", panel.flag_prr),
        ("bcpnn", panel.flag_bcpnn),
        ("mgps", panel.flag_mgps),
        ("three_of_four", panel.flag_three_of_four),
    ):
        counts[name] = int(np.sum(flag & sufficient))
        counts[f"{name}_including_insufficient"] = int(np.sum(flag))

    if panel.flag_all_four is None:
        counts["all_four"] = None
        counts["all_four_including_insufficient"] = None
    else:
        counts["all_four"] = int(np.sum(panel.flag_all_four & sufficient))
        counts["all_four_including_insufficient"] = int(np.sum(panel.flag_all_four))
    return counts


def _ic_divergence(panel: EstimatorPanel) -> dict[str, float]:
    """How far the locked closed-form IC sits from the exact posterior mean.

    Reported on every run because the two are not the same statistic and the gap
    grows as the expected count falls. See ``stats.bcpnn``.
    """
    finite = np.isfinite(panel.bcpnn.ic_observed_expected)
    if not bool(np.any(finite)):
        return {"max_absolute": 0.0, "median_absolute": 0.0, "share_above_one": 0.0}
    gap = np.abs(panel.bcpnn.ic_observed_expected[finite] - panel.bcpnn.ic[finite])
    return {
        "max_absolute": float(np.max(gap)),
        "median_absolute": float(np.median(gap)),
        "share_above_one": float(np.mean(gap > 1.0)),
    }


def build(
    spec: ContingencySpec | None = None,
    settings: Settings | None = None,
) -> tuple[SignalRun, Path]:
    """Build, score and persist one signal run. Returns the record and its path."""
    spec = spec or ContingencySpec()
    settings = settings or get_settings()
    started = time.monotonic()
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    pairs = build_contingency(spec, settings)
    # The gamma mixture reaches a bound on this corpus. The fit is accepted so
    # the columns exist and can be studied, and every consequence of that is
    # marked provisional: no EBGM number leaves this run as a measurement, and
    # flag_all_four is null rather than false.
    panel = estimate_all(pairs.table, min_a=spec.min_a, mgps_allow_boundary=True)

    frame = _rows(pairs, panel, run_id)
    partition = signal_root(settings) / f"run={run_id}"
    partition.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(partition / "part-0.parquet", compression="zstd")

    hyper = panel.mgps.hyperparameters
    record = SignalRun(
        run_id=run_id,
        created_at=datetime.now(UTC).isoformat(),
        params=spec.as_params(),
        code_sha=_code_sha(),
        data_snapshot=_data_snapshot(),
        n_cases=pairs.n_cases,
        pairs_observed=pairs.pairs_observed,
        pairs_sufficient=pairs.pairs_sufficient,
        duplicates_removed=pairs.duplicates_removed,
        cases_without_drug=pairs.cases_without_drug,
        cases_without_reaction=pairs.cases_without_reaction,
        hyperparameters={
            "alpha1": hyper.alpha1,
            "beta1": hyper.beta1,
            "alpha2": hyper.alpha2,
            "beta2": hyper.beta2,
            "p": hyper.p,
            "neg_log_likelihood": hyper.neg_log_likelihood,
            "iterations": hyper.iterations,
            "n_pairs_fitted": hyper.n_pairs,
            "n_rows_after_squashing": hyper.n_fitted_rows,
            "on_boundary": list(hyper.on_boundary),
        },
        mgps_provisional=hyper.provisional,
        flag_counts=_flag_counts(panel),
        ic_divergence=_ic_divergence(panel),
        seconds=time.monotonic() - started,
        peak_rss_bytes=_peak_rss_bytes(),
    )

    run_partition = run_root(settings) / f"run={run_id}"
    run_partition.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "run_id": [run_id],
            "created_at": [record.created_at],
            "code_sha": [record.code_sha],
            "params": [str(record.params)],
            "data_snapshot": [str(record.data_snapshot)],
            "n_cases": [record.n_cases],
            "pairs_written": [record.pairs_observed],
            "seconds": [record.seconds],
            "peak_rss_bytes": [record.peak_rss_bytes],
        }
    ).write_parquet(run_partition / "part-0.parquet", compression="zstd")

    (run_partition / "record.json").write_text(
        json.dumps(record.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    log.info(
        "signals.built",
        run_id=run_id,
        pairs=record.pairs_observed,
        three_of_four=record.flag_counts["three_of_four"],
        mgps_provisional=record.mgps_provisional,
        seconds=round(record.seconds, 1),
        peak_rss_gib=round(record.peak_rss_bytes / 1024**3, 2),
    )
    return record, partition


def latest_run(settings: Settings | None = None) -> str | None:
    """The most recent ``run_id`` on disk, or ``None`` if nothing has been built."""
    root = signal_root(settings)
    if not root.exists():
        return None
    runs = sorted(path.name.removeprefix("run=") for path in root.iterdir() if path.is_dir())
    return runs[-1] if runs else None


def read_run(run_id: str, settings: Settings | None = None) -> pl.DataFrame:
    """Every scored pair from one run."""
    partition = signal_root(settings) / f"run={run_id}"
    if not partition.exists():
        message = f"no signal run {run_id!r} under {signal_root(settings)}"
        raise FileNotFoundError(message)
    return pl.read_parquet(partition / "*.parquet")


#: Repository root, from this module's location: analytics -> signaldesk -> src.
REPO_ROOT = Path(__file__).resolve().parents[3]


def history_root() -> Path:
    """Where committed metric snapshots live. Repository data, not configurable."""
    return REPO_ROOT / "evals" / "history"


def read_record(run_id: str, settings: Settings | None = None) -> dict[str, object]:
    """The full provenance record for one run."""
    path = run_root(settings) / f"run={run_id}" / "record.json"
    if not path.exists():
        message = f"no run record at {path}"
        raise FileNotFoundError(message)
    document: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
    return document


def _is_provisional(run: dict[str, object]) -> bool:
    """Whether a run's MGPS scores were produced by a boundary fit."""
    mgps = run.get("mgps")
    return isinstance(mgps, dict) and bool(mgps.get("provisional"))


def write_history_artifact(
    run_ids: Sequence[str],
    settings: Settings | None = None,
    *,
    root: Path | None = None,
    diagnostics: Path | None = None,
) -> Path:
    """Collect the named runs into one committed artifact.

    One file, not one per run: the primary analysis and its sensitivity run are
    read together or not at all, and splitting them invites a number being
    quoted from one while the caption describes the other.
    """
    if not run_ids:
        message = "at least one run is needed to write an artifact"
        raise ValueError(message)

    runs = [read_record(run_id, settings) for run_id in run_ids]
    provisional = [run["run_id"] for run in runs if _is_provisional(run)]
    diagnostic = (
        json.loads(diagnostics.read_text(encoding="utf-8")) if diagnostics is not None else None
    )

    document = {
        "artifact": "signals",
        "written_at": datetime.now(UTC).isoformat(),
        "code_sha": _code_sha(),
        "runs": runs,
        "reference_validation": {
            "status": "blocked",
            "reason": (
                "evals/reference_sets/ is empty. The Harpaz, OMOP and EU-ADR standards "
                "and the outcome-to-PT map are curated by hand and are not generated by "
                "this project. No AUROC, sensitivity or specificity is reported until "
                "they exist."
            ),
            "required_files": [
                "evals/reference_sets/outcome_pt_map.yaml",
                "evals/reference_sets/harpaz_2013.csv",
                "evals/reference_sets/omop.csv",
                "evals/reference_sets/euadr.csv",
            ],
        },
        "mgps_boundary_diagnostic": diagnostic,
        "quotable": {
            "note": (
                "Numbers safe to quote elsewhere. Anything absent from this list is "
                "either provisional or not yet measured."
            ),
            "estimators": ["ror", "prr", "bcpnn"],
            "withheld": ["mgps"] if provisional else [],
            "withheld_reason": (
                "The gamma mixture prior converged onto a bound on this corpus. The "
                "profile likelihood shows the bound is the genuine optimum rather than "
                "an optimizer failure, but EBGM05 moves by 17.9 percent in flagged "
                "count across that profile, so which value is believed changes the "
                "reported number. EBGM and EBGM05 columns exist in the signal table "
                "and are not measurements."
            )
            if provisional
            else None,
        },
    }

    directory = root or history_root()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"signals_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log.info("signals.artifact.written", path=str(path), runs=len(runs))
    return path
