"""MGPS against openEBGM 0.9.1 on the CAERS data that package ships.

Three layers, because a single end-to-end comparison would not say which part
was wrong when it failed:

1. the likelihood, evaluated at openEBGM's own fitted hyperparameters;
2. the fit, run from this project's start points on openEBGM's own data;
3. the posterior scores - Qn, EBGM, EBGM05 - given those hyperparameters.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from signaldesk.core.errors import EstimatorConvergenceError
from signaldesk.stats.mgps import (
    fit_hyperparameters,
    gamma_poisson_shrinker,
    mixture_fraction,
    negative_log_likelihood,
    posterior_geometric_mean,
    posterior_percentile,
    profile_likelihood,
    squash,
)
from signaldesk.stats.types import Contingency, MgpsHyperparameters

pytestmark = pytest.mark.unit


def _theta(reference_mgps: dict[str, Any]) -> tuple[float, float, float, float, float]:
    hat = reference_mgps["theta_hat"]
    return (hat["alpha1"], hat["beta1"], hat["alpha2"], hat["beta2"], hat["p"])


def test_weighted_likelihood_matches_openebgm(reference_mgps: dict[str, Any]) -> None:
    """negLLsquash, on the squashed table the published fit was performed on."""
    squashed = reference_mgps["squashed"]
    value = negative_log_likelihood(
        _theta(reference_mgps),
        np.asarray(squashed["N"], dtype=np.float64),
        np.asarray(squashed["E"], dtype=np.float64),
        weights=np.asarray(squashed["weight"], dtype=np.float64),
    )
    assert value == pytest.approx(reference_mgps["neg_log_likelihood_squashed"], rel=1e-10)


def test_unweighted_likelihood_matches_openebgm(reference_mgps: dict[str, Any]) -> None:
    """negLL, unweighted, which is the form the corpus pass uses."""
    sample = reference_mgps["sample"]
    value = negative_log_likelihood(
        _theta(reference_mgps),
        np.asarray(sample["N"], dtype=np.float64),
        np.asarray(sample["E"], dtype=np.float64),
    )
    assert value == pytest.approx(reference_mgps["neg_log_likelihood_sample"], rel=1e-10)


def test_fit_reproduces_the_published_hyperparameters(reference_mgps: dict[str, Any]) -> None:
    """The optimizer finds openEBGM's optimum from this project's start points.

    openEBGM minimises with ``nlminb`` and this project with L-BFGS-B, from a
    different set of starting values, so agreement here is evidence about the
    likelihood surface rather than about one optimizer reproducing another.
    """
    squashed = reference_mgps["squashed"]
    fitted = fit_hyperparameters(
        np.asarray(squashed["N"], dtype=np.float64),
        np.asarray(squashed["E"], dtype=np.float64),
        weights=np.asarray(squashed["weight"], dtype=np.float64),
    )
    published = reference_mgps["theta_hat"]

    assert fitted.alpha1 == pytest.approx(published["alpha1"], rel=1e-3)
    assert fitted.beta1 == pytest.approx(published["beta1"], rel=1e-3)
    assert fitted.alpha2 == pytest.approx(published["alpha2"], rel=1e-3)
    assert fitted.beta2 == pytest.approx(published["beta2"], rel=1e-3)
    assert fitted.p == pytest.approx(published["p"], rel=1e-3)
    assert fitted.neg_log_likelihood == pytest.approx(
        reference_mgps["neg_log_likelihood_squashed"], rel=1e-8
    )


def test_mixture_fraction_matches_openebgm(reference_mgps: dict[str, Any]) -> None:
    sample = reference_mgps["sample"]
    qn = mixture_fraction(
        _theta(reference_mgps),
        np.asarray(sample["N"], dtype=np.float64),
        np.asarray(sample["E"], dtype=np.float64),
    )
    assert np.allclose(qn, np.asarray(sample["qn"], dtype=np.float64), rtol=1e-10)


def test_ebgm_matches_openebgm(reference_mgps: dict[str, Any]) -> None:
    """Per-pair EBGM over 300 pairs spanning the whole table.

    Checked against the published ``(N, E)`` directly rather than through a
    reconstructed 2x2, because rounding cells back to whole cases would perturb
    ``E`` in the fourth decimal and the comparison would then be testing the
    reconstruction rather than the estimator.

    openEBGM rounds its output to the requested number of digits, so the
    tolerance is set by that rounding, not by the arithmetic.
    """
    sample = reference_mgps["sample"]
    count = np.asarray(sample["N"], dtype=np.float64)
    expected = np.asarray(sample["E"], dtype=np.float64)
    theta = _theta(reference_mgps)
    qn = mixture_fraction(theta, count, expected)

    assert np.allclose(
        posterior_geometric_mean(theta, count, expected, qn),
        np.asarray(sample["ebgm"], dtype=np.float64),
        atol=1e-6,
    )


def test_ebgm05_matches_openebgm(reference_mgps: dict[str, Any]) -> None:
    """The 5th percentile of the posterior, found by bisection.

    openEBGM brackets from -100000 and this project from zero upward. Agreement
    to the published rounding is evidence about the percentile, not about the
    bracketing strategy.
    """
    sample = reference_mgps["sample"]
    count = np.asarray(sample["N"], dtype=np.float64)
    expected = np.asarray(sample["E"], dtype=np.float64)
    theta = _theta(reference_mgps)
    qn = mixture_fraction(theta, count, expected)

    assert np.allclose(
        posterior_percentile(theta, count, expected, qn, percentile=5.0),
        np.asarray(sample["ebgm05"], dtype=np.float64),
        atol=1e-6,
    )


def test_shrinker_wires_the_table_through_to_the_scored_functions() -> None:
    """``gamma_poisson_shrinker`` must add nothing to the validated functions.

    The reference comparisons above run on ``(N, E)`` arrays. This is the
    separate check that the table entry point derives ``E`` from the cells and
    then defers, so the two cannot drift apart.
    """
    table = Contingency(
        a=np.array([3, 25, 100, 4000], dtype=np.int64),
        b=np.array([97, 1975, 100, 46000], dtype=np.int64),
        c=np.array([500, 1000, 100, 90000], dtype=np.int64),
        d=np.array([99400, 97000, 99700, 9860000], dtype=np.int64),
    )
    published = MgpsHyperparameters(
        alpha1=2.94073934,
        beta1=0.38537165,
        alpha2=2.15410622,
        beta2=2.03677734,
        p=0.0720271218954894,
        neg_log_likelihood=0.0,
        iterations=0,
        n_pairs=4,
    )
    theta = published.as_tuple()
    result = gamma_poisson_shrinker(table, hyperparameters=published)

    count = table.a.astype(np.float64)
    expected = table.expected
    qn = mixture_fraction(theta, count, expected)

    assert np.allclose(result.qn, qn, rtol=0, atol=0)
    assert np.allclose(result.ebgm, posterior_geometric_mean(theta, count, expected, qn))
    assert np.allclose(result.ebgm05, posterior_percentile(theta, count, expected, qn))
    assert np.all(result.ebgm05 <= result.ebgm)


def test_ebgm05_never_exceeds_ebgm(reference_mgps: dict[str, Any]) -> None:
    sample = reference_mgps["sample"]
    assert np.all(
        np.asarray(sample["ebgm05"], dtype=np.float64)
        <= np.asarray(sample["ebgm"], dtype=np.float64)
    )


def test_empty_table_raises_rather_than_returning_a_default() -> None:
    with pytest.raises(EstimatorConvergenceError, match="empty table"):
        fit_hyperparameters(np.array([], dtype=np.float64), np.array([], dtype=np.float64))


def test_fitting_on_a_count_truncated_table_destroys_the_prior(
    reference_mgps: dict[str, Any],
) -> None:
    """Regression: the prior must not be fitted on the ``a >= 3`` subset.

    The first mixture component models the low-count mass - the pairs reported
    once or twice, which are most of the table. Remove them and the fit is not
    slightly off, it is a different prior: on this data the second component's
    shape goes from 2.15 to 316 and the mixture weight from 0.07 to 0.75, and
    every EBGM in the corpus would then be shrunk against it.

    On the FAERS corpus the same truncation drove the first shape onto its lower
    bound and raised instead, which is the louder half of the same failure. The
    quiet half is what this pins: the optimizer converges, reports success, and
    returns something unusable.

    This is why ``analytics.contingency`` returns every observed pair and the
    minimum cell count is applied as a flag by the estimator panel rather than
    as a filter before the fit.
    """
    squashed = reference_mgps["squashed"]
    count = np.asarray(squashed["N"], dtype=np.float64)
    expected = np.asarray(squashed["E"], dtype=np.float64)
    weights = np.asarray(squashed["weight"], dtype=np.float64)

    keep = count >= 3
    assert 0 < keep.sum() < count.size, "the fixture must contain low-count rows"

    whole = fit_hyperparameters(count, expected, weights=weights)
    truncated = fit_hyperparameters(count[keep], expected[keep], weights=weights[keep])

    assert whole.alpha2 == pytest.approx(reference_mgps["theta_hat"]["alpha2"], rel=1e-3)
    assert truncated.alpha2 > 100 * whole.alpha2
    assert truncated.p > 10 * whole.p


def test_a_fit_that_cannot_move_off_its_bound_raises() -> None:
    """Degenerate data must raise, not return an unshrunk ratio.

    Every pair having exactly its expected count leaves the mixture weight and
    the second component unidentified. The contract is that this surfaces as an
    error rather than as a plausible-looking EBGM column.
    """
    count = np.ones(50, dtype=np.float64)
    expected = np.ones(50, dtype=np.float64)
    with pytest.raises(EstimatorConvergenceError):
        fit_hyperparameters(count, expected)


def test_boundary_fit_is_returned_only_when_explicitly_allowed() -> None:
    """The opt-in must record which parameters are pinned, not merely permit them.

    ``on_boundary`` is what lets a caller mark the resulting scores provisional.
    A fit accepted without it recorded would be indistinguishable from a clean
    one at every point downstream.
    """
    count = np.ones(50, dtype=np.float64)
    expected = np.ones(50, dtype=np.float64)

    with pytest.raises(EstimatorConvergenceError, match="allow_boundary"):
        fit_hyperparameters(count, expected)

    fitted = fit_hyperparameters(count, expected, allow_boundary=True)

    assert fitted.on_boundary
    assert fitted.provisional


def test_an_interior_fit_reports_no_boundary(reference_mgps: dict[str, Any]) -> None:
    squashed = reference_mgps["squashed"]
    fitted = fit_hyperparameters(
        np.asarray(squashed["N"], dtype=np.float64),
        np.asarray(squashed["E"], dtype=np.float64),
        weights=np.asarray(squashed["weight"], dtype=np.float64),
    )

    assert fitted.on_boundary == ()
    assert not fitted.provisional


def test_squash_preserves_the_table_it_stands_for(reference_mgps: dict[str, Any]) -> None:
    """Total weight is the row count and every stratum keeps its count value.

    A squash that lost or invented weight would silently reweight the
    likelihood, which no downstream check would catch.
    """
    table = reference_mgps["table"]
    count = np.asarray(table["N"], dtype=np.float64)
    expected = np.asarray(table["E"], dtype=np.float64)

    squashed_n, squashed_e, weights = squash(count, expected)

    assert weights.sum() == pytest.approx(float(count.size))
    assert set(np.unique(squashed_n)) == set(np.unique(count))
    assert squashed_n.size < count.size
    assert np.all(squashed_e > 0)

    # Weight within each count stratum matches the number of pairs in it.
    for value in np.unique(count):
        assert weights[squashed_n == value].sum() == pytest.approx(
            float(np.count_nonzero(count == value))
        )


def test_squash_leaves_small_strata_whole(reference_mgps: dict[str, Any]) -> None:
    """Only strata with enough rows to bin are binned.

    Measured on this data: squashing the 16,040-pair N=1 stratum costs 0.3
    percent on the fitted prior, while also squashing the 710-pair N=2 stratum
    moves alpha1 from 2.94 to 22.8. The guard is what separates the two.
    """
    table = reference_mgps["table"]
    count = np.asarray(table["N"], dtype=np.float64)
    expected = np.asarray(table["E"], dtype=np.float64)

    squashed_n, _e, weights = squash(count, expected)

    large = count == 1
    small = count == 2
    assert np.count_nonzero(large) > 15_000
    assert 0 < np.count_nonzero(small) < 1_000

    assert np.count_nonzero(squashed_n == 1) < np.count_nonzero(large)
    assert np.count_nonzero(squashed_n == 2) == np.count_nonzero(small)
    assert np.all(weights[squashed_n == 2] == 1.0)


def test_fitting_the_squashed_table_reproduces_the_published_prior(
    reference_mgps: dict[str, Any],
) -> None:
    """The point of squashing: the same prior, from a table a thousand times smaller.

    openEBGM squashed twice and fitted with nlminb; this squashes once, with an
    independent implementation, and fits with L-BFGS-B. Agreement is evidence
    that the reduction preserves what the likelihood depends on.
    """
    table = reference_mgps["table"]
    count = np.asarray(table["N"], dtype=np.float64)
    expected = np.asarray(table["E"], dtype=np.float64)

    fitted = fit_hyperparameters(count, expected, squash_above=0)
    published = reference_mgps["theta_hat"]

    assert fitted.n_pairs == count.size
    assert fitted.n_fitted_rows < count.size / 10
    for name in ("alpha1", "beta1", "alpha2", "beta2", "p"):
        assert getattr(fitted, name) == pytest.approx(published[name], rel=0.005), name


def test_a_small_table_is_fitted_without_squashing(reference_mgps: dict[str, Any]) -> None:
    """Squashing is for tables too large to fit directly, and nothing else."""
    squashed = reference_mgps["squashed"]
    fitted = fit_hyperparameters(
        np.asarray(squashed["N"], dtype=np.float64),
        np.asarray(squashed["E"], dtype=np.float64),
        weights=np.asarray(squashed["weight"], dtype=np.float64),
    )

    assert fitted.n_fitted_rows == fitted.n_pairs


def test_profile_likelihood_fixes_its_parameter_and_refits_the_rest(
    reference_mgps: dict[str, Any],
) -> None:
    """Each point holds the fixed value and a theta that reproduces its own NLL.

    A profile whose reported likelihood did not correspond to its reported
    parameters would read as a curve while describing nothing.
    """
    squashed = reference_mgps["squashed"]
    count = np.asarray(squashed["N"], dtype=np.float64)
    expected = np.asarray(squashed["E"], dtype=np.float64)
    weights = np.asarray(squashed["weight"], dtype=np.float64)

    grid = (0.5, 1.0, 2.0, 4.0)
    profile = profile_likelihood(count, expected, parameter="alpha1", grid=grid, weights=weights)

    assert [point.value for point in profile] == list(grid)
    for point in profile:
        assert point.theta[0] == point.value
        assert negative_log_likelihood(
            point.theta, count, expected, weights=weights
        ) == pytest.approx(point.neg_log_likelihood, rel=1e-9)


def test_the_profile_minimum_agrees_with_the_free_fit(reference_mgps: dict[str, Any]) -> None:
    """A grid point at the fitted value must not beat the unconstrained optimum."""
    squashed = reference_mgps["squashed"]
    count = np.asarray(squashed["N"], dtype=np.float64)
    expected = np.asarray(squashed["E"], dtype=np.float64)
    weights = np.asarray(squashed["weight"], dtype=np.float64)
    published = reference_mgps["theta_hat"]["alpha1"]

    profile = profile_likelihood(
        count, expected, parameter="alpha1", grid=(published,), weights=weights
    )

    assert len(profile) == 1
    assert (
        profile[0].neg_log_likelihood
        >= reference_mgps["neg_log_likelihood_squashed"]
        - abs(reference_mgps["neg_log_likelihood_squashed"]) * 1e-6
    )


def test_profiling_an_unknown_parameter_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown parameter"):
        profile_likelihood(np.ones(10), np.ones(10), parameter="lambda", grid=(1.0,))
