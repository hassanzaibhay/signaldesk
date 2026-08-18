"""The four locked thresholds, and the conservative flag built from them."""

from __future__ import annotations

import numpy as np
import pytest

from signaldesk.stats.panel import MIN_CELL_A, estimate_all
from signaldesk.stats.types import Contingency, MgpsHyperparameters

pytestmark = pytest.mark.unit

# Fitted by openEBGM on its CAERS data; see tests/fixtures/estimators/. Supplied
# rather than refitted so these tests exercise the thresholds, not the optimizer.
PRIOR = MgpsHyperparameters(
    alpha1=2.94073934,
    beta1=0.38537165,
    alpha2=2.15410622,
    beta2=2.03677734,
    p=0.0720271218954894,
    neg_log_likelihood=0.0,
    iterations=0,
    n_pairs=0,
)


def _table() -> Contingency:
    # Row 0: strong on every method. Row 1: two co-reports, below the minimum
    # cell. Row 2: fewer co-reports than expected, so no method should flag it.
    return Contingency(
        a=np.array([400, 2, 5], dtype=np.int64),
        b=np.array([4_000, 98, 9_995], dtype=np.int64),
        c=np.array([2_000, 500, 100_000], dtype=np.int64),
        d=np.array([9_993_600, 9_999_400, 9_890_000], dtype=np.int64),
    )


def test_all_four_flags_agree_on_a_strong_pair() -> None:
    panel = estimate_all(_table(), hyperparameters=PRIOR)

    assert bool(panel.flag_ror[0])
    assert bool(panel.flag_prr[0])
    assert bool(panel.flag_bcpnn[0])
    assert bool(panel.flag_mgps[0])
    assert bool(panel.flag_all_four[0])


def test_no_flag_survives_on_an_under_reported_pair() -> None:
    panel = estimate_all(_table(), hyperparameters=PRIOR)

    assert not bool(panel.flag_all_four[2])
    assert float(panel.ror.ror[2]) < 1.0
    assert float(panel.bcpnn.ic025[2]) <= 0.0


def test_pairs_below_the_minimum_cell_are_computed_and_marked_not_dropped() -> None:
    """They stay in the table so the denominator downstream stays honest."""
    panel = estimate_all(_table(), hyperparameters=PRIOR)

    assert len(panel.contingency) == 3
    assert panel.insufficient.tolist() == [False, True, False]
    assert np.isfinite(panel.ror.ror[1])
    assert not bool(panel.flag_ror[1]), "ROR requires a >= 3 regardless of the bound"


def test_flag_all_four_is_the_conjunction_and_nothing_else() -> None:
    panel = estimate_all(_table(), hyperparameters=PRIOR)
    expected = panel.flag_ror & panel.flag_prr & panel.flag_bcpnn & panel.flag_mgps

    assert panel.flag_all_four.tolist() == expected.tolist()


def test_minimum_cell_is_the_locked_three() -> None:
    assert MIN_CELL_A == 3


def test_cells_sum_to_the_background_on_every_row() -> None:
    table = _table()
    assert table.n.tolist() == [10_000_000, 10_000_000, 10_000_000]


def test_a_provisional_prior_nulls_the_four_method_flag() -> None:
    """A headline count must not rest on a prior that is an artefact of a bound.

    ``None`` rather than an array of ``False``: absent and negative are
    different answers, and a column of ``False`` reads as "no pair met all
    four" to anything that consumes it.
    """
    provisional = MgpsHyperparameters(
        alpha1=1e-6,
        beta1=0.38537165,
        alpha2=2.15410622,
        beta2=2.03677734,
        p=0.0720271218954894,
        neg_log_likelihood=0.0,
        iterations=0,
        n_pairs=0,
        on_boundary=("alpha1",),
    )
    panel = estimate_all(_table(), hyperparameters=provisional)

    assert panel.mgps_provisional
    assert panel.flag_all_four is None
    assert (
        panel.flag_ror_prr_bcpnn.tolist()
        == (panel.flag_ror & panel.flag_prr & panel.flag_bcpnn).tolist()
    )


def test_an_interior_prior_reports_all_four() -> None:
    panel = estimate_all(_table(), hyperparameters=PRIOR)

    assert not panel.mgps_provisional
    assert panel.flag_all_four is not None
    assert panel.flag_all_four.tolist() == (panel.flag_ror_prr_bcpnn & panel.flag_mgps).tolist()


def test_an_adjudicated_boundary_prior_reports_all_four() -> None:
    """A boundary ruled on by a profile likelihood is reportable.

    flag_all_four was null because the prior was an artefact of the feasible
    region. Once a profile has settled that the bound is the optimum and that the
    scores do not move across it, the four-method count rests on something.
    """
    provisional = MgpsHyperparameters(
        alpha1=1e-6,
        beta1=0.38537165,
        alpha2=2.15410622,
        beta2=2.03677734,
        p=0.0720271218954894,
        neg_log_likelihood=0.0,
        iterations=0,
        n_pairs=0,
        on_boundary=("alpha1",),
    )
    panel = estimate_all(_table(), hyperparameters=provisional, mgps_boundary_adjudicated=True)

    assert not panel.mgps_provisional
    assert panel.mgps_boundary_adjudicated
    assert panel.flag_all_four is not None
    assert panel.flag_all_four.tolist() == (panel.flag_ror_prr_bcpnn & panel.flag_mgps).tolist()
    # The measurement behind the ruling is untouched: a pinned parameter stays
    # recorded as pinned whatever was decided about reporting it.
    assert panel.mgps.hyperparameters.on_boundary == ("alpha1",)
    assert panel.mgps.hyperparameters.provisional


def test_an_unadjudicated_boundary_prior_still_withholds_all_four() -> None:
    provisional = MgpsHyperparameters(
        alpha1=1e-6,
        beta1=0.38537165,
        alpha2=2.15410622,
        beta2=2.03677734,
        p=0.0720271218954894,
        neg_log_likelihood=0.0,
        iterations=0,
        n_pairs=0,
        on_boundary=("alpha1",),
    )
    panel = estimate_all(_table(), hyperparameters=provisional)

    assert panel.mgps_provisional
    assert not panel.mgps_boundary_adjudicated
    assert panel.flag_all_four is None
