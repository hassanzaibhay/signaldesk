"""Result types for the disproportionality estimators.

Frozen dataclasses rather than tuples or dicts. A tuple of six float arrays
invites a caller to unpack them in the wrong order, and nothing catches it: the
shapes match and the arithmetic downstream is still valid, just wrong. A named
field is checked at the point of use.

Every estimator returns arrays, not scalars. The pipeline computes millions of
pairs at once and a scalar API would force a Python loop over them, which is the
one thing the statistics layer is not allowed to do.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

#: Counts as they arrive from the contingency builder.
CountArray = NDArray[np.int64]

#: Everything the estimators produce.
FloatArray = NDArray[np.float64]

#: Per-pair booleans: correction applied, threshold met, count sufficient.
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class Contingency:
    """The 2x2 table for a set of drug-event pairs.

    ::

                    PT present    PT absent
        drug            a             b
        not drug        c             d

    The unit is the deduplicated case. ``a + b + c + d`` is the size of the
    analysis background for every row, which is checked on construction: a row
    whose cells do not sum to the corpus total means the marginals and the
    background were computed over different populations, and every statistic
    derived from it would be wrong in a way no downstream check would catch.
    """

    a: CountArray
    b: CountArray
    c: CountArray
    d: CountArray

    def __post_init__(self) -> None:
        shapes = {self.a.shape, self.b.shape, self.c.shape, self.d.shape}
        if len(shapes) != 1:
            message = f"cell arrays must share a shape, got {sorted(map(str, shapes))}"
            raise ValueError(message)
        for name, cell in (("a", self.a), ("b", self.b), ("c", self.c), ("d", self.d)):
            if cell.size and cell.min() < 0:
                message = f"negative count in cell {name}"
                raise ValueError(message)

    def __len__(self) -> int:
        return int(self.a.shape[0]) if self.a.ndim else 0

    @property
    def n(self) -> CountArray:
        """Total cases in the background, per row."""
        return self.a + self.b + self.c + self.d

    @property
    def n_drug(self) -> CountArray:
        """Cases naming the drug."""
        return self.a + self.b

    @property
    def n_event(self) -> CountArray:
        """Cases carrying the term."""
        return self.a + self.c

    @property
    def expected(self) -> FloatArray:
        """Expected co-occurrence count under independence, ``n_drug*n_event/N``.

        This is MGPS's ``E`` and BCPNN's denominator. Computed once here so the
        two cannot disagree about it.
        """
        return (self.n_drug.astype(np.float64) * self.n_event.astype(np.float64)) / self.n.astype(
            np.float64
        )

    @property
    def has_zero_cell(self) -> BoolArray:
        """Rows with at least one empty cell, where the ratio estimators break."""
        empty = (self.a == 0) | (self.b == 0) | (self.c == 0) | (self.d == 0)
        return np.asarray(empty, dtype=np.bool_)


@dataclass(frozen=True, slots=True)
class RorResult:
    """Reporting odds ratio with a log-normal 95 percent interval."""

    ror: FloatArray
    ror_lower: FloatArray
    ror_upper: FloatArray
    #: Haldane-Anscombe was applied to this row. See ``stats.correction``.
    corrected: BoolArray


@dataclass(frozen=True, slots=True)
class PrrResult:
    """Proportional reporting ratio, its interval, and the Yates chi-squared.

    ``chi2`` is computed on the uncorrected counts even where the ratio and its
    interval used corrected ones; see ``stats.prr`` for why.
    """

    prr: FloatArray
    prr_lower: FloatArray
    prr_upper: FloatArray
    chi2: FloatArray
    corrected: BoolArray


@dataclass(frozen=True, slots=True)
class BcpnnResult:
    """Information component, with the posterior it was drawn from.

    ``ic`` is the posterior mean under the Bate (1998) prior and ``ic025`` its
    ``ic - 2*sd`` lower bound, per the ARCHITECTURE 8.2 amendment of 2026-08-15.
    ``ic025_exact`` is the posterior's own 2.5th percentile, which differs only
    by the ``2`` against ``1.96`` multiplier.

    ``ic_observed_expected`` is the unshrunk ``log2(N*a/((a+b)(a+c)))`` that the
    section previously specified. It is carried so the divergence between a
    shrunk and an unshrunk statistic stays visible on every row rather than
    being resolved out of sight.
    """

    ic: FloatArray
    ic025: FloatArray
    ic_variance: FloatArray
    ic_observed_expected: FloatArray
    ic025_exact: FloatArray


@dataclass(frozen=True, slots=True)
class MgpsHyperparameters:
    """The fitted prior of the gamma-Poisson shrinker.

    A two-component gamma mixture: weight ``p`` on ``Gamma(alpha1, beta1)`` and
    ``1 - p`` on ``Gamma(alpha2, beta2)``, both in shape-rate form.
    """

    alpha1: float
    beta1: float
    alpha2: float
    beta2: float
    p: float
    #: Negative log-likelihood at the optimum, for the run artifact.
    neg_log_likelihood: float
    #: Optimizer iterations actually taken.
    iterations: int
    #: Pairs in the table the prior was fitted from.
    n_pairs: int
    #: Rows the likelihood was actually evaluated on. Smaller than ``n_pairs``
    #: when the table was squashed into weighted representative points.
    n_fitted_rows: int = 0
    #: Parameters sitting on a bound at the optimum. Empty for a clean interior
    #: fit. Non-empty means the fit was accepted under ``allow_boundary`` and
    #: every score derived from it is provisional.
    on_boundary: tuple[str, ...] = ()

    @property
    def provisional(self) -> bool:
        """Whether scores from this prior may be quoted as measurements."""
        return bool(self.on_boundary)

    def as_tuple(self) -> tuple[float, float, float, float, float]:
        """Parameters in the order the likelihood and posterior functions want."""
        return (self.alpha1, self.beta1, self.alpha2, self.beta2, self.p)


@dataclass(frozen=True, slots=True)
class MgpsResult:
    """Empirical Bayes geometric mean and its 5th percentile."""

    ebgm: FloatArray
    ebgm05: FloatArray
    #: Posterior probability the pair's rate came from the first mixture
    #: component, given its count. DuMouchel's Qn.
    qn: FloatArray
    hyperparameters: MgpsHyperparameters


@dataclass(frozen=True, slots=True)
class EstimatorPanel:
    """All four estimators over one contingency table, plus the locked flags."""

    contingency: Contingency
    ror: RorResult
    prr: PrrResult
    bcpnn: BcpnnResult
    mgps: MgpsResult

    flag_ror: BoolArray
    flag_prr: BoolArray
    flag_bcpnn: BoolArray
    flag_mgps: BoolArray
    #: ROR, PRR and BCPNN together. Reportable whatever MGPS did.
    flag_three_of_four: BoolArray
    #: The conservative definition, or ``None`` when the MGPS prior is
    #: provisional. ``None`` rather than an array of ``False``: absent and
    #: negative are different answers, and a column of ``False`` would be
    #: counted as "no pair met all four" by anything that consumed it.
    flag_all_four: BoolArray | None
    #: The gamma mixture prior sat on a bound. Every EBGM and EBGM05 on this
    #: panel is provisional and must not be quoted as a measurement.
    mgps_provisional: bool
    #: ``a`` below the minimum cell count. Computed, reported, but excluded from
    #: headline counts rather than dropped, so the denominator stays visible.
    insufficient: BoolArray
