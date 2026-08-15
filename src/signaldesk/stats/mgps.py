"""Multi-item gamma Poisson shrinker - EBGM and EBGM05.

DuMouchel (1999), "Bayesian Data Mining in Large Frequency Tables, With an
Application to the FDA Spontaneous Reporting System", The American Statistician
53(3):177-190, with the estimation refinements of DuMouchel and Pregibon (2001).

## Model

For a pair with expected count ``E`` under independence, the observed count is
``N ~ Poisson(E * lambda)``, and the relative reporting rate ``lambda`` is drawn
from a two-component gamma mixture that is estimated from the whole table::

    lambda ~ p * Gamma(alpha1, beta1) + (1 - p) * Gamma(alpha2, beta2)

in shape-rate form. Marginalising the Poisson over each gamma gives a negative
binomial, so the marginal likelihood of a count is a two-component negative
binomial mixture with ``size = alpha`` and ``prob = beta / (beta + E)``.

The five hyperparameters are fitted once, by maximum likelihood over every pair
in the table. That is the "empirical" in empirical Bayes: the prior is not
assumed, it is what the corpus as a whole says about how much a reporting rate
usually deviates from its expectation, and it is what shrinks a 3-out-of-3
pair back toward the null while leaving a 3000-case pair alone.

## Zero counts

Only pairs that were reported at least once exist in the table; the full cross
product of drugs and terms is billions of cells that are almost all zero and is
never materialised. The likelihood therefore conditions on ``N >= N*``, with
``N* = 1``. Each mixture component is truncated separately and the truncated
components are then mixed::

    f*(n) = f(n) / (1 - F(N* - 1))    per component
    L(n)  = p * f1*(n) + (1 - p) * f2*(n)

This is the ordering in ``openEBGM::negLL`` (CRAN 0.9.1) and is what the
published hyperparameter estimates are conditioned on. Truncating the mixture as
a whole instead would give a different, incompatible fit.

## Posterior

``Qn`` is the posterior probability the pair's rate came from the first
component. The posterior for ``lambda`` is then a two-component gamma mixture
with shapes ``alpha + N`` and rates ``beta + E``, weighted by ``Qn``::

    EBGM   = exp(E[ln lambda])              DuMouchel Eq. 9-11
    EBGM05 = 5th percentile of the posterior

The locked signal threshold is ``EBGM05 > 2`` (ARCHITECTURE section 8.2).

## Failure

A fit that does not converge raises ``EstimatorConvergenceError``. It does not
fall back to the raw observed-to-expected ratio, and it does not return NaN. An
unshrunk ratio presented in an EBGM column is a fabricated number, and the whole
reason this estimator is in the panel is the shrinkage.

Checked against ``openEBGM`` 0.9.1 on its shipped CAERS data: the likelihood at
a given hyperparameter vector, the fitted vector itself, and the per-pair
``Qn``, ``EBGM`` and ``EBGM05``. See ``tests/unit/stats/test_mgps.py``.
"""

from __future__ import annotations

import numpy as np

# SciPy ships no type information; every value it returns is narrowed to a
# concrete array type at the point of return below.
from scipy.optimize import minimize  # type: ignore[import-untyped]
from scipy.special import digamma, gammainc, gammaln  # type: ignore[import-untyped]

from signaldesk.core.errors import EstimatorConvergenceError
from signaldesk.stats.types import (
    BoolArray,
    Contingency,
    FloatArray,
    MgpsHyperparameters,
    MgpsResult,
)

#: Smallest count the table holds. The contingency builder materialises only
#: pairs reported at least once, so the likelihood conditions on N >= 1.
N_STAR = 1

#: Starting points for the optimizer, tried in order and the best kept. Several
#: rather than one because the mixture likelihood is not convex and a single
#: start silently converges to a local optimum on some corpora. These are
#: DuMouchel's published starting values and the two used by openEBGM's own
#: examples; they are here in the source, not in configuration, so that a run
#: artifact naming this module's commit pins the fit exactly.
START_POINTS: tuple[tuple[float, float, float, float, float], ...] = (
    (0.2, 0.1, 2.0, 4.0, 1.0 / 3.0),
    (0.5, 0.5, 2.0, 2.0, 0.1),
    (1.0, 1.0, 3.0, 3.0, 0.2),
    (2.0, 0.5, 2.0, 2.0, 0.1),
)

#: Bounds. Shapes and rates strictly positive; the mixture weight strictly
#: inside the unit interval, because either endpoint collapses the model to one
#: component and leaves the other two parameters unidentified.
BOUNDS: tuple[tuple[float, float], ...] = (
    (1e-6, 1e4),
    (1e-6, 1e4),
    (1e-6, 1e4),
    (1e-6, 1e4),
    (1e-6, 1.0 - 1e-6),
)

#: Rows per chunk when accumulating the likelihood. The optimizer evaluates it
#: dozens of times, so the per-evaluation temporaries, not the table itself, set
#: the memory peak.
CHUNK_ROWS = 1_000_000

#: Bisection steps for the posterior percentile. 100 halvings of the bracket is
#: far below float64 resolution at any plausible EBGM05.
BISECTION_STEPS = 100

#: Largest relative rate the bracket search will grow to before giving up.
BISECTION_CEILING = 1e18


def _log_negative_binomial(
    count: FloatArray, shape: float, rate: float, expected: FloatArray
) -> FloatArray:
    """Log pmf of the count, having marginalised the Poisson over one gamma.

    ``size = shape`` and ``prob = rate / (rate + E)`` in the usual negative
    binomial parameterisation.
    """
    log_prob = np.log(rate) - np.log(rate + expected)
    log_one_minus = np.log(expected) - np.log(rate + expected)
    log_pmf = (
        gammaln(count + shape)
        - gammaln(shape)
        - gammaln(count + 1.0)
        + shape * log_prob
        + count * log_one_minus
    )
    return np.asarray(log_pmf, dtype=np.float64)


def _log_survival_at_zero(shape: float, rate: float, expected: FloatArray) -> FloatArray:
    """``log(1 - F(N* - 1))`` for one component, with ``N* = 1``.

    ``F(0)`` of a negative binomial is ``prob ** size``, so the survival is
    ``1 - (rate / (rate + E)) ** shape``. Computed with ``log1p``/``expm1`` so
    that a component putting almost all its mass on zero does not lose every
    significant digit.
    """
    log_f0 = shape * (np.log(rate) - np.log(rate + expected))
    return np.asarray(np.log(-np.expm1(log_f0)), dtype=np.float64)


def _log_mixture(
    theta: tuple[float, float, float, float, float],
    count: FloatArray,
    expected: FloatArray,
    *,
    truncate: bool,
) -> FloatArray:
    """Log of the (optionally zero-truncated) two-component mixture density."""
    alpha1, beta1, alpha2, beta2, p = theta
    log_f1 = _log_negative_binomial(count, alpha1, beta1, expected)
    log_f2 = _log_negative_binomial(count, alpha2, beta2, expected)
    if truncate:
        log_f1 = log_f1 - _log_survival_at_zero(alpha1, beta1, expected)
        log_f2 = log_f2 - _log_survival_at_zero(alpha2, beta2, expected)
    return np.asarray(np.logaddexp(np.log(p) + log_f1, np.log1p(-p) + log_f2), dtype=np.float64)


def negative_log_likelihood(
    theta: tuple[float, float, float, float, float],
    count: FloatArray,
    expected: FloatArray,
    *,
    weights: FloatArray | None = None,
) -> float:
    """Zero-truncated negative log-likelihood of the table, summed in chunks.

    ``weights`` supports data squashing, where one row stands for many pairs
    with similar ``(N, E)``. The production pipeline passes no weights; the
    reference fixture does, because the published estimates were produced from
    squashed data.
    """
    total = 0.0
    for start in range(0, count.shape[0], CHUNK_ROWS):
        stop = start + CHUNK_ROWS
        log_density = _log_mixture(theta, count[start:stop], expected[start:stop], truncate=True)
        if weights is None:
            total += float(-np.sum(log_density))
        else:
            total += float(-np.sum(weights[start:stop] * log_density))
    return total


def fit_hyperparameters(
    count: FloatArray,
    expected: FloatArray,
    *,
    weights: FloatArray | None = None,
    start_points: tuple[tuple[float, float, float, float, float], ...] = START_POINTS,
) -> MgpsHyperparameters:
    """Fit the five hyperparameters by maximum likelihood over the whole table.

    Every start point in ``start_points`` is run and the best optimum kept. A
    start that fails is skipped; if none succeeds, or if the winning optimum
    sits on a bound, ``EstimatorConvergenceError`` is raised. A parameter pinned
    to a bound means the likelihood wanted to leave the feasible region, so the
    reported value is an artefact of the box rather than an estimate.
    """
    if count.shape != expected.shape:
        message = f"count and expected must align, got {count.shape} and {expected.shape}"
        raise ValueError(message)
    if count.size == 0:
        message = "cannot fit the gamma mixture on an empty table"
        raise EstimatorConvergenceError(message)

    def objective(vector: FloatArray) -> float:
        theta = (
            float(vector[0]),
            float(vector[1]),
            float(vector[2]),
            float(vector[3]),
            float(vector[4]),
        )
        value = negative_log_likelihood(theta, count, expected, weights=weights)
        # L-BFGS-B cannot step out of a non-finite region, so an excursion into
        # one is reported as very bad rather than as NaN.
        return value if np.isfinite(value) else 1e300

    best: tuple[float, FloatArray, int] | None = None
    failures: list[str] = []
    for start in start_points:
        outcome = minimize(
            objective,
            np.asarray(start, dtype=np.float64),
            method="L-BFGS-B",
            bounds=BOUNDS,
        )
        if not outcome.success or not np.isfinite(outcome.fun):
            failures.append(f"start {start}: {outcome.message}")
            continue
        if best is None or outcome.fun < best[0]:
            best = (float(outcome.fun), np.asarray(outcome.x, dtype=np.float64), int(outcome.nit))

    if best is None:
        message = "gamma mixture fit did not converge from any start point: " + "; ".join(failures)
        raise EstimatorConvergenceError(message)

    value, vector, iterations = best
    for index, (parameter, (low, high)) in enumerate(zip(vector, BOUNDS, strict=True)):
        if parameter <= low * (1.0 + 1e-6) or parameter >= high * (1.0 - 1e-6):
            message = (
                f"gamma mixture parameter {index} converged onto its bound "
                f"({parameter:g} in [{low:g}, {high:g}]); the fit is an artefact of the "
                "feasible region, not an estimate"
            )
            raise EstimatorConvergenceError(message)

    return MgpsHyperparameters(
        alpha1=float(vector[0]),
        beta1=float(vector[1]),
        alpha2=float(vector[2]),
        beta2=float(vector[3]),
        p=float(vector[4]),
        neg_log_likelihood=value,
        iterations=iterations,
        n_pairs=int(count.shape[0]),
    )


def mixture_fraction(
    theta: tuple[float, float, float, float, float],
    count: FloatArray,
    expected: FloatArray,
) -> FloatArray:
    """``Qn``: posterior probability the rate came from the first component.

    Untruncated, unlike the likelihood: this conditions on the count that was
    actually observed, so the question of which counts could have been observed
    does not arise.
    """
    alpha1, beta1, alpha2, beta2, p = theta
    log_f1 = np.log(p) + _log_negative_binomial(count, alpha1, beta1, expected)
    log_f2 = np.log1p(-p) + _log_negative_binomial(count, alpha2, beta2, expected)
    return np.asarray(np.exp(log_f1 - np.logaddexp(log_f1, log_f2)), dtype=np.float64)


def posterior_geometric_mean(
    theta: tuple[float, float, float, float, float],
    count: FloatArray,
    expected: FloatArray,
    qn: FloatArray,
) -> FloatArray:
    """``EBGM = exp(E[ln lambda])`` under the posterior. DuMouchel Eq. 9 to 11.

    For ``X ~ Gamma(shape, rate)``, ``E[ln X] = digamma(shape) - ln(rate)``, and
    the posterior shape and rate are ``alpha + N`` and ``beta + E``. The mixture
    weight applies to the expectations, not to the exponentials: the geometric
    mean of the mixture is not the mixture of the geometric means.
    """
    alpha1, beta1, alpha2, beta2, _ = theta
    log_lambda = qn * (digamma(alpha1 + count) - np.log(beta1 + expected)) + (1.0 - qn) * (
        digamma(alpha2 + count) - np.log(beta2 + expected)
    )
    return np.asarray(np.exp(log_lambda), dtype=np.float64)


def _posterior_cdf(
    theta: tuple[float, float, float, float, float],
    count: FloatArray,
    expected: FloatArray,
    qn: FloatArray,
    point: FloatArray,
) -> FloatArray:
    """Posterior CDF of ``lambda`` at ``point``, per pair."""
    alpha1, beta1, alpha2, beta2, _ = theta
    first = gammainc(alpha1 + count, (beta1 + expected) * point)
    second = gammainc(alpha2 + count, (beta2 + expected) * point)
    return np.asarray(qn * first + (1.0 - qn) * second, dtype=np.float64)


def posterior_percentile(
    theta: tuple[float, float, float, float, float],
    count: FloatArray,
    expected: FloatArray,
    qn: FloatArray,
    *,
    percentile: float = 5.0,
) -> FloatArray:
    """Percentile of the posterior for each pair, by vectorized bisection.

    A closed form does not exist for a gamma mixture. Bisection is used rather
    than a per-row root finder because there are millions of rows and a Python
    loop over them is exactly what this layer forbids.
    """
    target = percentile / 100.0
    lower = np.zeros_like(count, dtype=np.float64)
    upper = np.ones_like(count, dtype=np.float64)

    # Grow the bracket until it contains the percentile for every pair.
    for _ in range(64):
        if bool(np.all(_posterior_cdf(theta, count, expected, qn, upper) >= target)):
            break
        upper = upper * 2.0
        if float(np.max(upper)) > BISECTION_CEILING:
            message = (
                f"posterior {percentile}th percentile exceeded {BISECTION_CEILING:g} for at "
                "least one pair; the fitted prior is degenerate"
            )
            raise EstimatorConvergenceError(message)
    else:  # pragma: no cover - the ceiling check raises first in practice
        message = "could not bracket the posterior percentile"
        raise EstimatorConvergenceError(message)

    for _ in range(BISECTION_STEPS):
        middle = 0.5 * (lower + upper)
        below = _posterior_cdf(theta, count, expected, qn, middle) < target
        lower = np.where(below, middle, lower)
        upper = np.where(below, upper, middle)

    return 0.5 * (lower + upper)


def gamma_poisson_shrinker(
    table: Contingency,
    *,
    hyperparameters: MgpsHyperparameters | None = None,
) -> MgpsResult:
    """EBGM and EBGM05 for every row, fitting the prior if one is not supplied.

    Pass ``hyperparameters`` to score a subset against a prior fitted on the
    whole corpus. Scoring a subset against a prior fitted on that same subset
    would let the shrinkage change with the filter, and two pages of the site
    would then disagree about one pair's EBGM.
    """
    count = table.a.astype(np.float64)
    expected = table.expected

    fitted = hyperparameters or fit_hyperparameters(count, expected)
    theta = fitted.as_tuple()

    qn = mixture_fraction(theta, count, expected)

    return MgpsResult(
        ebgm=posterior_geometric_mean(theta, count, expected, qn),
        ebgm05=posterior_percentile(theta, count, expected, qn, percentile=5.0),
        qn=qn,
        hyperparameters=fitted,
    )


def flag(result: MgpsResult, *, min_ebgm05: float = 2.0) -> BoolArray:
    """The locked MGPS signal rule: ``EBGM05 > 2``."""
    return np.asarray(result.ebgm05 > min_ebgm05, dtype=np.bool_)
