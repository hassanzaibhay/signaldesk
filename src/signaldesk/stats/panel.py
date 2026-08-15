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
) -> EstimatorPanel:
    """Run all four estimators and apply the locked thresholds.

    Pairs below ``min_a`` are computed rather than skipped, and marked
    ``insufficient``. Dropping them here would remove them from the denominator
    of every rate reported downstream while leaving nothing to say they existed.

    ``mgps_allow_boundary`` accepts a gamma mixture fit that sits on a bound
    rather than raising. ``panel.mgps_provisional`` is then true, and with it
    ``flag_all_four`` is **not** computed: a conservative four-method count built
    on a prior that is an artefact of the feasible region would be a headline
    number resting on an unvalidated one. ``flag_three_of_four`` is what such a
    run can report.
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
    flag_three = flag_ror & flag_prr & flag_bcpnn

    provisional = mgps_result.hyperparameters.provisional

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
        flag_three_of_four=flag_three,
        flag_all_four=None if provisional else flag_three & flag_mgps,
        mgps_provisional=provisional,
        insufficient=np.asarray(table.a < min_a, dtype=np.bool_),
    )
