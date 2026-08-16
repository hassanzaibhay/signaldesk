"""Is the MGPS boundary fit a limit of the data, or a failed search?

The gamma mixture prior converges onto the ``alpha1`` lower bound on the FAERS
corpus. That has two readings and they call for opposite actions:

* the zero-truncated negative binomial tends to the logarithmic distribution as
  its shape goes to zero, a legitimate limit for a table dominated by pairs
  reported exactly once; or
* the optimizer never reached the real optimum, and the estimator is defective.

Only the profile likelihood separates them, and the separation is one
comparison, fixed here before any number is produced:

    NLL(bound) <= min over the grid   ->  the bound is the genuine optimum
    NLL(bound) >  min over the grid   ->  the fit is wrong

Monotonicity of the curve is reported next to it as an observation about shape.
It does not decide anything on its own: a curve can wander far from the bound
while the bound is still the optimum.

The second question is separate and decides publishability rather than
correctness. Even granting the bound, if EBGM05 moves materially across the
profile then which point is believed changes the reported signal count, and no
MGPS number can be quoted until it is pinned. So EBGM05 is scored at the bound
**and** at the grid argmin without exception - those are the two points the
comparison turns on - and the spread between the extremes is reported.

The run parquet already persists ``a``, ``expected`` and ``insufficient``, so
this reads a completed run rather than rebuilding the contingency table.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np

from signaldesk.analytics.signals import read_run
from signaldesk.core.errors import EstimatorConvergenceError
from signaldesk.core.logging import get_logger
from signaldesk.stats.mgps import (
    BOUNDS,
    PARAMETER_NAMES,
    mixture_fraction,
    posterior_percentile,
    profile_likelihood,
    squash,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from signaldesk.core.config import Settings
    from signaldesk.stats.mgps import ProfilePoint
    from signaldesk.stats.types import BoolArray, FloatArray

log = get_logger(__name__)

#: Grid for ``alpha1``, strictly above its lower bound. Dense near the bound,
#: where the comparison that settles the reading is made, and sparse further out
#: where the curve only has to be described.
DEFAULT_GRID: tuple[float, ...] = (
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.35,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
    3.0,
)

#: Extra points to score EBGM05 at, for the shape of the sensitivity. The bound
#: and the grid argmin are always scored on top of these.
DEFAULT_SCORE_AT: tuple[float, ...] = (1e-5, 1e-3, 0.01, 0.1, 1.0, 3.0)

#: The signal threshold under test. Locked in ARCHITECTURE section 8.2.
MIN_EBGM05 = 2.0


def _point_as_dict(point: ProfilePoint) -> dict[str, Any]:
    """One profile point, with both sweeps kept separate."""
    return {
        "alpha1": point.value,
        "nll": point.neg_log_likelihood,
        "theta": list(point.theta),
        "forward_nll": point.forward_nll,
        "backward_nll": point.backward_nll,
        "sweeps_agree": point.sweeps_agree,
    }


def _score(
    theta: tuple[float, float, float, float, float],
    count: FloatArray,
    expected: FloatArray,
    sufficient: BoolArray,
) -> dict[str, Any]:
    """EBGM05 over the full table at one hyperparameter vector.

    Counted over ``sufficient`` only, the same denominator the run record uses
    for its headline flag counts, so the two are comparable without a caveat.
    """
    qn = mixture_fraction(theta, count, expected)
    ebgm05 = posterior_percentile(theta, count, expected, qn, percentile=5.0)
    kept = ebgm05[sufficient]
    return {
        "alpha1": theta[0],
        "theta": list(theta),
        "flagged_ebgm05_gt_2": int(np.sum(kept > MIN_EBGM05)),
        "median": float(np.median(kept)),
        "q90": float(np.percentile(kept, 90)),
        "q99": float(np.percentile(kept, 99)),
        "max": float(np.max(kept)),
    }


def build_diagnostic(
    run_id: str,
    settings: Settings | None = None,
    *,
    grid: Sequence[float] = DEFAULT_GRID,
    score_at: Sequence[float] = DEFAULT_SCORE_AT,
) -> dict[str, Any]:
    """Profile ``alpha1`` for one persisted run and score EBGM05 across it."""
    lower = float(BOUNDS[PARAMETER_NAMES.index("alpha1")][0])
    above = tuple(sorted({float(value) for value in grid if float(value) > lower}))
    if not above:
        message = f"the grid must hold at least one alpha1 above the lower bound {lower:g}"
        raise ValueError(message)

    frame = read_run(run_id, settings)
    count = frame["a"].to_numpy().astype(np.float64)
    expected = frame["expected"].to_numpy().astype(np.float64)
    sufficient = ~frame["insufficient"].to_numpy()
    log.info(
        "mgps.diagnostic.loaded",
        run_id=run_id,
        pairs=int(count.size),
        sufficient=int(sufficient.sum()),
    )

    # Squash once and reuse for every refit. The profile would otherwise squash
    # the same table again per call, and the squashing is what makes an 8.6
    # million pair fit finish at all.
    squashed_count, squashed_expected, weights = squash(count, expected)

    def report(direction: str, value: float, nll: float | None) -> None:
        """Progress for a command that otherwise runs silent for tens of minutes."""
        log.info(
            "mgps.diagnostic.point",
            run_id=run_id,
            direction=direction,
            alpha1=value,
            nll=nll,
            converged=nll is not None,
        )

    profile = profile_likelihood(
        squashed_count,
        squashed_expected,
        parameter="alpha1",
        grid=(lower, *above),
        weights=weights,
        on_point=report,
    )
    by_alpha = {point.value: point for point in profile}
    if lower not in by_alpha:
        message = "every refit at the alpha1 lower bound failed; the profile cannot be read"
        raise EstimatorConvergenceError(message)

    at_bound = by_alpha[lower]
    on_grid = [point for point in profile if point.value > lower]
    if not on_grid:
        message = "every refit above the alpha1 lower bound failed; the profile cannot be read"
        raise EstimatorConvergenceError(message)

    argmin = min(on_grid, key=lambda point: point.neg_log_likelihood)
    bound_is_optimum = at_bound.neg_log_likelihood <= argmin.neg_log_likelihood

    ordered = [at_bound, *on_grid]
    nlls = [point.neg_log_likelihood for point in ordered]
    monotone = all(nlls[i] <= nlls[i + 1] + 1e-6 for i in range(len(nlls) - 1))
    disagreeing = [point.value for point in ordered if not point.sweeps_agree]

    # The bound and the argmin are what the comparison turns on, so they are
    # scored whether or not the caller asked for them.
    wanted = sorted({lower, argmin.value, *(float(value) for value in score_at)})
    sensitivity = []
    for value in wanted:
        if value not in by_alpha:
            continue
        row = _score(by_alpha[value].theta, count, expected, sufficient)
        log.info(
            "mgps.diagnostic.scored",
            run_id=run_id,
            alpha1=value,
            flagged=row["flagged_ebgm05_gt_2"],
            median=row["median"],
        )
        sensitivity.append(row)
    flagged = [row["flagged_ebgm05_gt_2"] for row in sensitivity]
    low, high = min(flagged), max(flagged)

    return {
        "run_id": run_id,
        "parameter": "alpha1",
        "lower_bound": lower,
        "sufficient_pairs": int(sufficient.sum()),
        "pairs": int(count.size),
        "rows_after_squashing": int(squashed_count.size),
        "decision_rule": (
            "The bound is the genuine optimum when NLL at the bound is no worse "
            "than the minimum over the grid. Monotonicity is reported as an "
            "observation about shape and does not decide the reading."
        ),
        "nll_at_bound": at_bound.neg_log_likelihood,
        "grid_argmin": {
            "alpha1": argmin.value,
            "nll": argmin.neg_log_likelihood,
        },
        "bound_minus_argmin": at_bound.neg_log_likelihood - argmin.neg_log_likelihood,
        "bound_is_optimum": bound_is_optimum,
        "monotone_away_from_bound": monotone,
        "sweeps_agree_everywhere": not disagreeing,
        "sweeps_disagree_at": disagreeing,
        "profile": [_point_as_dict(point) for point in ordered],
        "ebgm05_sensitivity": sensitivity,
        "ebgm05_at_bound": next(row for row in sensitivity if row["alpha1"] == lower),
        "ebgm05_at_grid_argmin": next(row for row in sensitivity if row["alpha1"] == argmin.value),
        "flagged_minimum": low,
        "flagged_maximum": high,
        "relative_spread": (high - low) / low if low else None,
    }


def write_diagnostic(
    run_id: str,
    out: Path,
    settings: Settings | None = None,
    *,
    grid: Sequence[float] = DEFAULT_GRID,
    score_at: Sequence[float] = DEFAULT_SCORE_AT,
) -> dict[str, Any]:
    """Build the diagnostic for ``run_id`` and write it to ``out`` as JSON."""
    document = build_diagnostic(run_id, settings, grid=grid, score_at=score_at)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log.info(
        "mgps.diagnostic.written",
        run_id=run_id,
        path=str(out),
        bound_is_optimum=document["bound_is_optimum"],
        relative_spread=document["relative_spread"],
    )
    return document
