"""Bayesian confidence propagation neural network - the information component.

## The prior, and why this one

Bate et al. (1998) put independent Beta priors on the three probabilities the
information component is built from, and the posteriors are conjugate:

===================  ===========================  ==================================
Quantity             Prior                        Posterior after observing
===================  ===========================  ==================================
``p(drug)``          ``Beta(a1, a - a1)``         ``Beta(a1 + n1., a - a1 + N - n1.)``
``p(event)``         ``Beta(b1, b - b1)``         ``Beta(b1 + n.1, b - b1 + N - n.1)``
``p(drug, event)``   ``Beta(g11, g - g11)``       ``Beta(g11 + a, g - g11 + N - a)``
===================  ===========================  ==================================

with ``a1 = b1 = g11 = 1`` and ``a = b = 2``: a uniform prior on each marginal,
worth one prior case each way. ``g`` is not free. It is fixed by requiring the
prior to be centred on independence, ``E[p(drug, event)] = E[p(drug)]E[p(event)]``,
which gives::

    g = g11 * (N + a)(N + b) / ((n1. + a1)(n.1 + b1))

That is the whole content of the prior: no drug-event association is expected
before the data, and one prior case of evidence is what it takes to move it.
This is what makes the IC lower bound shrink toward 0 as counts fall, which is
the reason to run BCPNN alongside ROR rather than instead of it.

## Derivation of the variance

``IC = log2 p(drug, event) - log2 p(drug) - log2 p(event)``, a sum of three
independent log-Beta variables. For ``X ~ Beta(u, v)``::

    E[ln X]   = psi(u) - psi(u + v)
    Var[ln X] = psi'(u) - psi'(u + v)

so, dividing by ``ln 2`` once for the mean and squaring it for the variance, and
adding the three variances because the terms are treated as independent::

    E[IC] = (1/ln2)  [ (psi(r1) - psi(r1+r2)) - (psi(p1) - psi(p1+p2))
                                              - (psi(q1) - psi(q1+q2)) ]
    V[IC] = (1/ln2)^2[ (psi'(r1) - psi'(r1+r2)) + (psi'(p1) - psi'(p1+p2))
                                                + (psi'(q1) - psi'(q1+q2)) ]

    p1 = 1 + n1.   p1 + p2 = N + 2
    q1 = 1 + n.1   q1 + q2 = N + 2
    r1 = 1 + a     r1 + r2 = N + g

Every term is a log-Beta of the same form, which is the check that the three are
parallel: each is ``psi(first shape) - psi(sum of both shapes)`` of its own
posterior. This derivation reproduces ``PhViD::BCPNN`` (CRAN 1.0.8) exactly -
its ``r2b = N - n11 - 1 + (2+N)^2/(q1*p1)`` is ``g - g11 + N - a`` written out.

## What is reported

ARCHITECTURE section 8.2 locks the reported statistic to the closed form
``IC = log2(N*a / ((a+b)(a+c)))`` with ``IC025 = IC - 2*sqrt(V)``, the
Noren/WHO-UMC convention. That closed form is the observed-to-expected ratio,
not the posterior mean, so both are returned: ``ic``/``ic025`` are the locked
pair that the signal table and its threshold use, and ``ic_posterior_mean``/
``ic025_exact`` are the exact posterior mean and its 2.5th percentile.

**The two do not agree, and what separates them is the expected count, not the
observed one.** ``g`` grows without bound as ``E`` falls, so a pair between a
rare drug and a rare term carries an enormous prior and its posterior mean is
shrunk hard, while the closed form is not shrunk at all. Measured on the
reference tables in ``tests/fixtures/estimators/``:

=========  =========  ==========  ===================  ==============
``a``      ``E``      ``IC``      posterior mean       difference
=========  =========  ==========  ===================  ==============
5000       577.5      3.114       3.112                0.002
25         20.5       0.286       0.245                0.041
100        0.400      7.966       6.169                1.797
4          0.0002     14.025      2.268                11.757
=========  =========  ==========  ===================  ==============

The consequence is that ``ic025``, which subtracts a shrunk posterior's standard
deviation from an unshrunk point estimate, is far less conservative than the
posterior's own 2.5th percentile exactly where the evidence is thinnest. Both
columns are therefore written to every signal row, and the build artifact
records the corpus-wide divergence. Reconciling them is an ARCHITECTURE 8.2
amendment and is not decided here.

The closed form is ``-inf`` at ``a = 0`` where the posterior mean is finite.
Pairs with ``a = 0`` are not materialised by the contingency builder, so this
arises only in tests, and it is left as ``-inf`` rather than patched: the locked
statistic genuinely is undefined there, and the posterior column beside it says
what the Bayesian answer would have been.
"""

from __future__ import annotations

import numpy as np

# SciPy ships no type information; every value it returns is narrowed to a
# concrete array type at the point of return below.
from scipy.special import digamma, polygamma  # type: ignore[import-untyped]

from signaldesk.stats.types import BcpnnResult, BoolArray, Contingency, FloatArray

#: Prior pseudo-counts. Uniform on each marginal, centred on independence.
PRIOR_ALPHA_1 = 1.0
PRIOR_ALPHA = 2.0
PRIOR_BETA_1 = 1.0
PRIOR_BETA = 2.0
PRIOR_GAMMA_11 = 1.0

#: Standard deviations subtracted for the locked lower bound. Two, not 1.96:
#: ARCHITECTURE section 8.2 locks ``IC - 2*sqrt(V)``, which is the WHO-UMC
#: convention and is very slightly more conservative than the 2.5th percentile.
IC025_SIGMA = 2.0

#: Exact 2.5th percentile of the posterior, for the column reported beside it.
EXACT_SIGMA = 1.959963984540054

_LN2 = float(np.log(2.0))


def information_component(table: Contingency) -> BcpnnResult:
    """IC and IC025, with the exact posterior mean and percentile alongside."""
    a = table.a.astype(np.float64)
    n = table.n.astype(np.float64)
    n_drug = table.n_drug.astype(np.float64)
    n_event = table.n_event.astype(np.float64)

    gamma = (
        PRIOR_GAMMA_11
        * ((n + PRIOR_ALPHA) * (n + PRIOR_BETA))
        / ((n_drug + PRIOR_ALPHA_1) * (n_event + PRIOR_BETA_1))
    )

    p1 = PRIOR_ALPHA_1 + n_drug
    p_total = n + PRIOR_ALPHA
    q1 = PRIOR_BETA_1 + n_event
    q_total = n + PRIOR_BETA
    r1 = PRIOR_GAMMA_11 + a
    r_total = n + gamma

    posterior_mean = (
        (digamma(r1) - digamma(r_total))
        - (digamma(p1) - digamma(p_total))
        - (digamma(q1) - digamma(q_total))
    ) / _LN2

    variance = (
        (polygamma(1, r1) - polygamma(1, r_total))
        + (polygamma(1, p1) - polygamma(1, p_total))
        + (polygamma(1, q1) - polygamma(1, q_total))
    ) / (_LN2**2)

    standard_deviation = np.sqrt(np.clip(variance, 0.0, None))

    # a = 0 makes the ratio 0/E, which is -inf, and 0/0 where the drug or the
    # term appears in no case at all. Both are the same answer - no observed
    # co-reporting, so no signal - but the second arrives as a NaN if it is
    # evaluated, and a NaN in a statistics column is indistinguishable from a
    # measurement once it is written to disk. So the division is only taken
    # where it means something.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(a > 0.0, n * a / (n_drug * n_event), 0.0)
        ic = np.where(a > 0.0, np.log2(ratio), -np.inf)

    return BcpnnResult(
        ic=ic,
        ic025=ic - IC025_SIGMA * standard_deviation,
        ic_variance=variance,
        ic_posterior_mean=posterior_mean,
        ic025_exact=posterior_mean - EXACT_SIGMA * standard_deviation,
    )


def flag(result: BcpnnResult) -> BoolArray:
    """The locked BCPNN signal rule: ``IC025 > 0``."""
    return np.asarray(result.ic025 > 0.0, dtype=np.bool_)


def expected_counts(table: Contingency) -> FloatArray:
    """``n_drug * n_event / N``, the denominator of the closed-form IC."""
    return table.expected
