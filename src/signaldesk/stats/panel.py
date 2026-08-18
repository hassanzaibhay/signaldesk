"""All four estimators over one contingency table, with the locked flags.

The panel exists so that the four thresholds live in exactly one place. Every
consumer - the signal build, the reference-set validation, the API - reads
``flag_all_four`` from here rather than re-deriving it, because a threshold
duplicated at three call sites is a threshold that will eventually be three
different thresholds.

Thresholds are ARCHITECTURE section 8.2 and are not parameters of this function:

===========  ====================================
Method       Signal
===========  ====================================
ROR          ``a >= 3`` and ``ROR025 > 1``
PRR          ``chi2 >= 4`` and ``PRR025 > 1``
BCPNN        ``IC025 > 0``
MGPS         ``EBGM05 > 2``
===========  ====================================

``flag_all_four`` is the conservative definition used for headline counts.
"""

from __future__ import annotations

import numpy as np

from signaldesk.stats import bcpnn, mgps, prr, ror
from signaldesk.stats.types import Contingency, EstimatorPanel, MgpsHyperparameters

#: Minimum cases in cell ``a`` for a pair to enter the headline counts.
MIN_CELL_A = 3


def estimate_all(
    table: Contingency,
    *,
    min_a: int = MIN_CELL_A,
    hyperparameters: MgpsHyperparameters | None = None,
    mgps_allow_boundary: bool = False,
    mgps_boundary_adjudicated: bool = False,
) -> EstimatorPanel:
    """Run all four estimators and apply the locked thresholds.

    Pairs below ``min_a`` are computed rather than skipped, and marked
    ``insufficient``. Dropping them here would remove them from the denominator
    of every rate reported downstream while leaving nothing to say they existed.

    ``mgps_allow_boundary`` accepts a gamma mixture fit that sits on a bound
    rather than raising. ``panel.mgps_provisional`` is then true, and with it
    ``flag_all_four`` is **not** computed: a conservative four-method count built
    on a prior that is an artefact of the feasible region would be a headline
    number resting on an unvalidated one. ``flag_ror_prr_bcpnn`` is what such a
    run can report.

    ``mgps_boundary_adjudicated`` says that boundary has been ruled on, and the
    scores derived from it are no longer withheld: ``flag_all_four`` is computed.
    The ruling is made outside this function, on a profile likelihood - whether
    the bound is the optimum, and whether the scores move across the profile -
    because a fit cannot conclude anything about its own boundary from the inside.
    ``analytics.mgps_diagnostic`` produces that evidence.

    Adjudication changes only what is reported. ``hyperparameters.on_boundary``
    still names every pinned parameter: that one sat on a bound is a measured
    fact and stays recorded whatever the ruling.

    ``flag_ror_prr_bcpnn`` is the conjunction of those three estimators, not a
    k-of-n rule over all four. Where MGPS flags a pair the other three do not,
    "any three of the four agreed" is a strictly larger set, and the two must
    not be read as the same number.
    """
    ror_result = ror.reporting_odds_ratio(table)
    prr_result = prr.proportional_reporting_ratio(table)
    bcpnn_result = bcpnn.information_component(table)
    mgps_result = mgps.gamma_poisson_shrinker(
        table, hyperparameters=hyperparameters, allow_boundary=mgps_allow_boundary
    )

    flag_ror = ror.flag(table, ror_result, min_a=min_a)
    flag_prr = prr.flag(prr_result)
    flag_bcpnn = bcpnn.flag(bcpnn_result)
    flag_mgps = mgps.flag(mgps_result)
    flag_ror_prr_bcpnn = flag_ror & flag_prr & flag_bcpnn

    provisional = mgps_result.hyperparameters.provisional and not mgps_boundary_adjudicated

    return EstimatorPanel(
        contingency=table,
        ror=ror_result,
        prr=prr_result,
        bcpnn=bcpnn_result,
        mgps=mgps_result,
        flag_ror=flag_ror,
        flag_prr=flag_prr,
        flag_bcpnn=flag_bcpnn,
        flag_mgps=flag_mgps,
        flag_ror_prr_bcpnn=flag_ror_prr_bcpnn,
        flag_all_four=None if provisional else flag_ror_prr_bcpnn & flag_mgps,
        mgps_provisional=provisional,
        mgps_boundary_adjudicated=mgps_boundary_adjudicated,
        insufficient=np.asarray(table.a < min_a, dtype=np.bool_),
    )
