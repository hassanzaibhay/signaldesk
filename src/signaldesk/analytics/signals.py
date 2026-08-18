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
from signaldesk.stats.mgps import fit_hyperparameters
from signaldesk.stats.panel import estimate_all
from signaldesk.stats.types import EstimatorPanel, FloatArray

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
                    "flag_ror_prr_bcpnn is what this run supports."
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


def _rows(
    pairs: PairTable,
    panel: EstimatorPanel,
    run_id: str,
    *,
    start: int = 0,
    stop: int | None = None,
) -> pl.DataFrame:
    """The scored rows for ``pairs[start:stop]``, which ``panel`` must cover.

    ``start``/``stop`` select the label slice to pair with a panel scored over
    the same rows. They default to the whole table.
    """
    stop = len(pairs) if stop is None else stop
    table = panel.contingency
    width = stop - start
    return pl.DataFrame(
        {
            "run_id": [run_id] * width,
            "drug": pairs.drug[start:stop],
            "pt": pairs.pt[start:stop],
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
            "flag_ror_prr_bcpnn": panel.flag_ror_prr_bcpnn,
            "flag_all_four": (
                panel.flag_all_four if panel.flag_all_four is not None else [None] * width
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
        ("ror_prr_bcpnn", panel.flag_ror_prr_bcpnn),
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


def _ic_gap(panel: EstimatorPanel) -> FloatArray:
    """Absolute gap between the closed-form IC and the exact posterior mean."""
    finite = np.isfinite(panel.bcpnn.ic_observed_expected)
    return np.abs(panel.bcpnn.ic_observed_expected[finite] - panel.bcpnn.ic[finite])


def _ic_divergence_from_gap(gap: FloatArray) -> dict[str, float]:
    """How far the locked closed-form IC sits from the exact posterior mean.

    Reported on every run because the two are not the same statistic and the gap
    grows as the expected count falls. See ``stats.bcpnn``.

    Takes the gap rather than a panel so a chunked run can concatenate the gaps
    and get the same median as an unchunked one. A median cannot be accumulated
    from per-chunk medians, and averaging them would be a different number
    wearing this one's name.
    """
    if not gap.size:
        return {"max_absolute": 0.0, "median_absolute": 0.0, "share_above_one": 0.0}
    return {
        "max_absolute": float(np.max(gap)),
        "median_absolute": float(np.median(gap)),
        "share_above_one": float(np.mean(gap > 1.0)),
    }


def _ic_divergence(panel: EstimatorPanel) -> dict[str, float]:
    """``_ic_divergence_from_gap`` over a single panel."""
    return _ic_divergence_from_gap(_ic_gap(panel))


def _merge_counts(
    total: dict[str, int | None], chunk: dict[str, int | None]
) -> dict[str, int | None]:
    """Add one chunk's flag counts into the running total.

    ``None`` is contagious: ``all_four`` is ``None`` when the MGPS prior is
    provisional, and a total that turned that into a number would be reporting a
    count the run does not support.
    """
    if not total:
        return dict(chunk)
    merged: dict[str, int | None] = {}
    for key, value in chunk.items():
        running = total.get(key)
        merged[key] = None if running is None or value is None else running + value
    return merged


def build(
    spec: ContingencySpec | None = None,
    settings: Settings | None = None,
    *,
    chunk_rows: int | None = None,
    mgps_adjudicated: bool = False,
) -> tuple[SignalRun, Path]:
    """Build, score and persist one signal run. Returns the record and its path.

    ``chunk_rows`` scores and writes the table in row batches instead of all at
    once, trading a little wall clock for a much lower peak. The raw-string
    corpus peaks at 8.2 GiB of an 11.7 GiB container, so a key that yields more
    pairs - ``--drug-key ingredient`` - needs this rather than 3.5 GiB of
    headroom and hope. Output is identical either way: every statistic is per
    row, and the one quantity fitted across rows, the MGPS prior, is fitted once
    on the whole table below and passed into every chunk.
    """
    spec = spec or ContingencySpec()
    settings = settings or get_settings()
    started = time.monotonic()
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    pairs = build_contingency(spec, settings)
    partition = signal_root(settings) / f"run={run_id}"
    partition.mkdir(parents=True, exist_ok=True)

    total = len(pairs)
    step = max(1, chunk_rows if chunk_rows else total)
    # The gamma mixture reaches a bound on this corpus. The fit is accepted so
    # the columns exist and can be studied, and every consequence of that is
    # marked provisional: no EBGM number leaves this run as a measurement, and
    # flag_all_four is null rather than false.
    prior = fit_hyperparameters(
        pairs.table.a.astype(np.float64), pairs.table.expected, allow_boundary=True
    )

    flag_counts: dict[str, int | None] = {}
    gaps: list[FloatArray] = []
    panel: EstimatorPanel | None = None
    for index, start in enumerate(range(0, total, step) if total else [0]):
        stop = min(start + step, total)
        panel = estimate_all(
            pairs.table.rows(start, stop),
            min_a=spec.min_a,
            hyperparameters=prior,
            mgps_allow_boundary=True,
            mgps_boundary_adjudicated=mgps_adjudicated,
        )
        _rows(pairs, panel, run_id, start=start, stop=stop).write_parquet(
            partition / f"part-{index}.parquet", compression="zstd"
        )
        flag_counts = _merge_counts(flag_counts, _flag_counts(panel))
        gaps.append(_ic_gap(panel))
        if step < total:
            log.info("signals.chunk.scored", run_id=run_id, rows=stop, of=total, chunk=index)

    if panel is None:  # pragma: no cover - build_contingency never returns nothing
        message = f"contingency for {spec.as_params()} produced no pairs"
        raise ValueError(message)

    hyper = prior
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
        mgps_provisional=panel.mgps_provisional,
        flag_counts=flag_counts,
        ic_divergence=_ic_divergence_from_gap(
            np.concatenate(gaps) if gaps else np.asarray([], dtype=np.float64)
        ),
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
        ror_prr_bcpnn=record.flag_counts["ror_prr_bcpnn"],
        chunks=len(gaps),
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


#: ``flag_three_of_four`` was renamed to ``flag_ror_prr_bcpnn``: the old name
#: reads as "any three of the four agreed", which is a different and larger set
#: wherever MGPS flags a pair the other three do not. Runs written before the
#: rename are read through these aliases rather than rebuilt, since the data and
#: the arithmetic behind them are unchanged. The first build after the rename
#: writes the new name natively and these can go.
LEGACY_FLAG_COLUMN = "flag_three_of_four"
LEGACY_FLAG_COLUMNS = {LEGACY_FLAG_COLUMN: "flag_ror_prr_bcpnn"}
LEGACY_FLAG_COUNT_KEYS = {
    "three_of_four": "ror_prr_bcpnn",
    "three_of_four_including_insufficient": "ror_prr_bcpnn_including_insufficient",
}


def read_run(run_id: str, settings: Settings | None = None) -> pl.DataFrame:
    """Every scored pair from one run, under current column names."""
    partition = signal_root(settings) / f"run={run_id}"
    if not partition.exists():
        message = f"no signal run {run_id!r} under {signal_root(settings)}"
        raise FileNotFoundError(message)
    frame = pl.read_parquet(partition / "*.parquet")
    renames = {
        old: new
        for old, new in LEGACY_FLAG_COLUMNS.items()
        if old in frame.columns and new not in frame.columns
    }
    return frame.rename(renames) if renames else frame


#: Repository root, from this module's location: analytics -> signaldesk -> src.
REPO_ROOT = Path(__file__).resolve().parents[3]


def history_root() -> Path:
    """Where committed metric snapshots live. Repository data, not configurable."""
    return REPO_ROOT / "evals" / "history"


def read_record(run_id: str, settings: Settings | None = None) -> dict[str, object]:
    """The full provenance record for one run, under current key names.

    This is the path that reaches the committed artifact, whose ``runs`` block is
    the record verbatim, so the legacy flag-count keys are mapped here too.
    """
    path = run_root(settings) / f"run={run_id}" / "record.json"
    if not path.exists():
        message = f"no run record at {path}"
        raise FileNotFoundError(message)
    document: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
    counts = document.get("flag_counts")
    if isinstance(counts, dict):
        document["flag_counts"] = {
            LEGACY_FLAG_COUNT_KEYS.get(key, key): value for key, value in counts.items()
        }
    return document


def _uses_legacy_flag_name(run_ids: Sequence[str], settings: Settings | None = None) -> bool:
    """Whether any included run's parquet still carries the pre-rename column."""
    for run_id in run_ids:
        partition = signal_root(settings) / f"run={run_id}"
        parts = sorted(partition.glob("*.parquet")) if partition.exists() else []
        if parts and LEGACY_FLAG_COLUMN in pl.read_parquet_schema(parts[0]):
            return True
    return False


def _is_provisional(run: dict[str, object]) -> bool:
    """Whether a run's MGPS scores were produced by a boundary fit."""
    mgps = run.get("mgps")
    return isinstance(mgps, dict) and bool(mgps.get("provisional"))


def _complete_all_four(run: dict[str, object], settings: Settings | None = None) -> bool:
    """Fill in ``all_four`` for a run built while MGPS was withheld.

    The run wrote null because a four-method count resting on an unadjudicated
    prior would have been a headline number resting on an unvalidated one. The
    prior has since been adjudicated, and the count is exact from what the run
    already persisted: ``flag_ror_prr_bcpnn & flag_mgps`` per row, headline over
    the pairs at or above the minimum cell count.

    Recomputing here rather than rebuilding the run keeps the run immutable and
    keeps its ``run_id`` - which the committed diagnostic names - intact. Returns
    whether anything was filled in, so the artifact can say so.
    """
    counts = run.get("flag_counts")
    if not isinstance(counts, dict) or counts.get("all_four") is not None:
        return False

    frame = read_run(str(run["run_id"]), settings)
    both = frame["flag_ror_prr_bcpnn"].to_numpy() & frame["flag_mgps"].to_numpy()
    counts["all_four"] = int(np.sum(both & ~frame["insufficient"].to_numpy()))
    counts["all_four_including_insufficient"] = int(np.sum(both))
    return True


def _withheld_reason(diagnostic: dict[str, object] | None) -> str:
    """Why MGPS is withheld, stating only what was actually measured.

    The spread is read from the diagnostic rather than written here. A constant
    in this file would be reported for every future run whatever that run
    measured, and would sit in the same document as a diagnostic contradicting
    it.
    """
    boundary = (
        "The gamma mixture prior converged onto a bound on this corpus. "
        "EBGM and EBGM05 columns exist in the signal table and are not "
        "measurements."
    )
    if diagnostic is None:
        return (
            f"{boundary} Whether that bound is the genuine optimum, and how far "
            "EBGM05 moves across the profile likelihood, has not been measured "
            "for this run."
        )

    spread = diagnostic.get("relative_spread")
    movement = (
        f"EBGM05 moves by {float(spread) * 100:.1f} percent in flagged count "
        "across that profile, so which value is believed changes the reported "
        "number."
        if isinstance(spread, int | float)
        else "The EBGM05 sensitivity across that profile was not recorded."
    )
    if diagnostic.get("bound_is_optimum"):
        return (
            f"{boundary} The profile likelihood shows the bound is the genuine "
            f"optimum rather than an optimizer failure, but {movement}"
        )
    return (
        f"{boundary} The profile likelihood finds a lower value away from the "
        f"bound, so the fit is defective rather than describing a limit of the "
        f"data. {movement}"
    )


def _flagged_at(diagnostic: dict[str, object], key: str) -> int | None:
    """The flagged count from one scored point of the diagnostic."""
    point = diagnostic.get(key)
    if not isinstance(point, dict):
        return None
    value = point.get("flagged_ebgm05_gt_2")
    return int(value) if isinstance(value, int) else None


def _mgps_adjudication(diagnostic: dict[str, object]) -> dict[str, object]:
    """The recorded basis for accepting a boundary fit.

    Every figure is read from the diagnostic rather than written here, for the
    same reason the withheld spread is: a constant in this file would be
    published for every future run whatever that run measured.

    The sensitivity figure is the movement between the bound and the grid
    argmin. Those are the two points the reading turns on, and they are where the
    likelihood has support. The full-profile spread is recorded next to it as
    context, with the point driving it and how far the likelihood puts that point
    from the bound, because a spread measured across values the data excludes is
    not a sensitivity result.
    """
    bound = _flagged_at(diagnostic, "ebgm05_at_bound")
    argmin = _flagged_at(diagnostic, "ebgm05_at_grid_argmin")
    delta = abs(argmin - bound) if bound is not None and argmin is not None else None

    profile = diagnostic.get("profile")
    at_bound = diagnostic.get("nll_at_bound")
    sensitivity = diagnostic.get("ebgm05_sensitivity")
    driver: dict[str, object] | None = None
    if (
        isinstance(profile, list)
        and isinstance(at_bound, int | float)
        and isinstance(sensitivity, list)
    ):
        scored = [
            row
            for row in sensitivity
            if isinstance(row, dict) and isinstance(row.get("flagged_ebgm05_gt_2"), int)
        ]
        worst = max(
            scored,
            key=lambda row: int(row["flagged_ebgm05_gt_2"]),
            default=None,
        )
        if worst is not None:
            point = next(
                (p for p in profile if isinstance(p, dict) and p.get("alpha1") == worst["alpha1"]),
                None,
            )
            driver = {
                "alpha1": worst["alpha1"],
                "flagged_ebgm05_gt_2": int(worst["flagged_ebgm05_gt_2"]),
                "nll_excess_over_bound": (
                    float(point["nll"]) - float(at_bound)
                    if isinstance(point, dict) and isinstance(point.get("nll"), int | float)
                    else None
                ),
            }

    return {
        "adjudicated": True,
        "criterion": (
            "Whether EBGM05 moves materially across the profile, fixed before the "
            "diagnostic ran. The sensitivity figure is the movement between the "
            "bound and the grid argmin; the full-profile spread is context."
        ),
        "bound_is_optimum": diagnostic.get("bound_is_optimum"),
        "monotone_away_from_bound": diagnostic.get("monotone_away_from_bound"),
        "near_bound_sensitivity": {
            "bound_flagged": bound,
            "grid_argmin_flagged": argmin,
            "flagged_delta": delta,
            "flagged_delta_relative_to_bound": (
                delta / bound if delta is not None and bound else None
            ),
        },
        "full_profile_context": {
            "relative_spread": diagnostic.get("relative_spread"),
            "relative_spread_denominator": "flagged_minimum",
            "flagged_minimum": diagnostic.get("flagged_minimum"),
            "flagged_maximum": diagnostic.get("flagged_maximum"),
            "driven_by": driver,
            "note": (
                "A spread measured across grid points the likelihood rejects is "
                "not a sensitivity result. It is recorded so the near-bound "
                "figure can be read against it."
            ),
        },
    }


def write_history_artifact(
    run_ids: Sequence[str],
    settings: Settings | None = None,
    *,
    root: Path | None = None,
    diagnostics: Path | None = None,
    mgps_adjudicated: bool = False,
) -> Path:
    """Collect the named runs into one committed artifact.

    One file, not one per run: the primary analysis and its sensitivity run are
    read together or not at all, and splitting them invites a number being
    quoted from one while the caption describes the other.

    ``mgps_adjudicated`` lifts the withholding on a boundary fit. It is an
    explicit input and is never inferred from the diagnostic: whether a measured
    sensitivity is small enough to report on is a judgement, not something the
    numbers decide. What the numbers do decide is whether the judgement is
    available to make, so this refuses without a diagnostic, and refuses on a
    diagnostic saying the bound is not the optimum.
    """
    if not run_ids:
        message = "at least one run is needed to write an artifact"
        raise ValueError(message)

    runs = [read_record(run_id, settings) for run_id in run_ids]
    loaded: dict[str, object] | None = (
        json.loads(diagnostics.read_text(encoding="utf-8")) if diagnostics is not None else None
    )
    diagnostic: dict[str, object] | None = loaded if isinstance(loaded, dict) else None

    if mgps_adjudicated:
        if diagnostic is None:
            message = (
                "mgps_adjudicated needs the diagnostic it rests on; pass the profile "
                "likelihood written by 'signaldesk signals mgps-diagnostic'"
            )
            raise ValueError(message)
        if not diagnostic.get("bound_is_optimum"):
            message = (
                "the diagnostic reports the bound is not the optimum, so the fit is "
                "defective and cannot be adjudicated as reportable"
            )
            raise ValueError(message)

    completed = [
        run["run_id"] for run in runs if mgps_adjudicated and _complete_all_four(run, settings)
    ]
    if mgps_adjudicated:
        for run in runs:
            mgps = run.get("mgps")
            if isinstance(mgps, dict) and mgps.get("provisional"):
                # Policy text, regenerated. The measured facts beside it -
                # provisional, on_boundary, the hyperparameters - are the run's
                # own record and are left exactly as it wrote them.
                mgps["note"] = (
                    "The gamma mixture prior converged onto a bound. The profile "
                    "likelihood recorded in this artifact settles that the bound "
                    "is the optimum and that EBGM05 does not move materially "
                    "across it, so EBGM and EBGM05 are reportable and "
                    "flag_all_four is computed. See mgps_adjudication."
                )
    provisional = [run["run_id"] for run in runs if not mgps_adjudicated and _is_provisional(run)]

    shas = sorted({str(run.get("code_sha", "unknown")) for run in runs})
    unrecorded = [run["run_id"] for run in runs if str(run.get("code_sha", "unknown")) == "unknown"]

    document = {
        "artifact": "signals",
        "written_at": datetime.now(UTC).isoformat(),
        # The commit the artifact was assembled at, which is not the provenance
        # of the numbers in it. Each run carries its own code_sha and that is
        # what produced its counts.
        "writer_code_sha": _code_sha(),
        "runs_code_sha": {
            "distinct": shas,
            "runs_at_unrecorded_commit": unrecorded,
            "note": (
                "Every included run recorded the commit it was built at."
                if not unrecorded
                else (
                    "These runs were produced at an unrecorded commit. "
                    "writer_code_sha describes only the process that assembled "
                    "this file and is not the provenance of their numbers."
                )
            ),
        },
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
        "mgps_adjudication": (
            _mgps_adjudication(diagnostic) if mgps_adjudicated and diagnostic else None
        ),
        "quotable": {
            "note": (
                "Numbers safe to quote elsewhere. Anything absent from this list is "
                "either provisional or not yet measured."
            ),
            "estimators": ["ror", "prr", "bcpnn"]
            if provisional
            else ["ror", "prr", "bcpnn", "mgps"],
            "withheld": ["mgps"] if provisional else [],
            "withheld_reason": _withheld_reason(diagnostic) if provisional else None,
        },
    }

    if completed:
        document["flag_all_four_note"] = (
            "These runs wrote flag_all_four as null, because they were built "
            "while the MGPS boundary was unadjudicated. The count here is "
            "recomputed from what each run persisted - flag_ror_prr_bcpnn and "
            "flag_mgps per row, headline over pairs at or above the minimum cell "
            "count - rather than by rebuilding, so the runs stay immutable and "
            "keep the run_id the diagnostic names. The flag_all_four column in "
            "their parquet is still null; the next build writes it natively. "
            f"Completed here for: {', '.join(str(run) for run in completed)}."
        )

    if _uses_legacy_flag_name(run_ids, settings):
        document["schema_note"] = (
            "flag_three_of_four was renamed to flag_ror_prr_bcpnn, which states "
            "the definition: the conjunction of ROR, PRR and BCPNN, not 'any "
            "three of the four agreed'. The run parquet for these runs still "
            "carries the old column name; this artifact carries the new one. The "
            "counts are unchanged - the rename is applied when the run is read, "
            "and the runs were not rebuilt."
        )

    directory = root or history_root()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"signals_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log.info("signals.artifact.written", path=str(path), runs=len(runs))
    return path
