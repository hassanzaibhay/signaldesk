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

import platform
import resource
import subprocess
import time
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
    hyperparameters: dict[str, float | int]
    flag_counts: dict[str, int]
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
            "mgps_hyperparameters": self.hyperparameters,
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

    An empty string here would look like a value. ``unknown`` says the
    provenance is incomplete, which is a different and worse thing than a run
    that has not been committed yet.
    """
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
            "ic_posterior_mean": panel.bcpnn.ic_posterior_mean,
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
            "flag_all_four": panel.flag_all_four,
        }
    )


def _ic_divergence(panel: EstimatorPanel) -> dict[str, float]:
    """How far the locked closed-form IC sits from the exact posterior mean.

    Reported on every run because the two are not the same statistic and the gap
    grows as the expected count falls. See ``stats.bcpnn``.
    """
    finite = np.isfinite(panel.bcpnn.ic)
    if not bool(np.any(finite)):
        return {"max_absolute": 0.0, "median_absolute": 0.0, "share_above_one": 0.0}
    gap = np.abs(panel.bcpnn.ic[finite] - panel.bcpnn.ic_posterior_mean[finite])
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
    panel = estimate_all(pairs.table, min_a=spec.min_a)

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
        },
        flag_counts={
            "ror": int(np.sum(panel.flag_ror)),
            "prr": int(np.sum(panel.flag_prr)),
            "bcpnn": int(np.sum(panel.flag_bcpnn)),
            "mgps": int(np.sum(panel.flag_mgps)),
            "all_four": int(np.sum(panel.flag_all_four)),
            "insufficient": int(np.sum(panel.insufficient)),
            "haldane_anscombe_corrected": int(np.sum(panel.ror.corrected)),
        },
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

    log.info(
        "signals.built",
        run_id=run_id,
        pairs=record.pairs_observed,
        all_four=record.flag_counts["all_four"],
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
