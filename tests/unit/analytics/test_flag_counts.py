"""Headline flag counts exclude pairs below the minimum cell count.

Regression. ARCHITECTURE section 8.1 says pairs below ``a >= 3`` are computed
and flagged but excluded from headline counts. Only the ROR rule carries that
minimum in its own definition, so on the full corpus PRR alone flagged
4,039,013 pairs against 2,785,896 pairs with three or more cases: a headline
count of signals larger than the number of pairs eligible to be signals.
"""

from __future__ import annotations

import numpy as np
import pytest

from signaldesk.analytics.signals import _flag_counts
from signaldesk.stats.panel import estimate_all
from signaldesk.stats.types import Contingency, MgpsHyperparameters

pytestmark = pytest.mark.unit

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
    # Rows 0 and 1 are strong and sufficient. Rows 2 and 3 are strong on the
    # ratio but carry one and two cases, so they are below the minimum.
    return Contingency(
        a=np.array([400, 50, 1, 2], dtype=np.int64),
        b=np.array([4_000, 500, 10, 20], dtype=np.int64),
        c=np.array([2_000, 2_000, 2_000, 2_000], dtype=np.int64),
        d=np.array([9_993_600, 9_997_450, 9_997_989, 9_997_978], dtype=np.int64),
    )


def test_headline_counts_exclude_insufficient_pairs() -> None:
    panel = estimate_all(_table(), hyperparameters=PRIOR)
    counts = _flag_counts(panel)

    assert counts["insufficient"] == 2
    for name in ("ror", "prr", "bcpnn", "three_of_four"):
        headline = counts[name]
        raw = counts[f"{name}_including_insufficient"]
        assert headline is not None and raw is not None
        assert headline <= raw


def test_a_rule_without_a_minimum_term_still_gets_a_bounded_headline() -> None:
    """PRR has no ``a >= 3`` term, so its raw count can exceed the eligible set.

    This is the shape of the corpus defect, reproduced small: the raw PRR count
    includes pairs the minimum excludes, and the headline count does not.
    """
    table = _table()
    panel = estimate_all(table, hyperparameters=PRIOR)
    counts = _flag_counts(panel)
    sufficient = int(np.count_nonzero(table.a >= 3))

    assert counts["prr_including_insufficient"] > sufficient
    assert counts["prr"] is not None
    assert counts["prr"] <= sufficient


def test_ror_headline_and_raw_agree_because_its_rule_carries_the_minimum() -> None:
    counts = _flag_counts(estimate_all(_table(), hyperparameters=PRIOR))

    assert counts["ror"] == counts["ror_including_insufficient"]


def test_a_provisional_prior_leaves_both_four_method_counts_null() -> None:
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
    counts = _flag_counts(estimate_all(_table(), hyperparameters=provisional))

    assert counts["all_four"] is None
    assert counts["all_four_including_insufficient"] is None
    assert counts["three_of_four"] is not None


def test_the_history_root_is_inside_the_repository() -> None:
    """Regression: it resolved to the filesystem root and failed on write.

    ``signals.py`` sits one directory shallower than the eval modules, so the
    same parents[] expression copied between them points somewhere different.
    """
    from signaldesk.analytics.signals import history_root
    from signaldesk.evals.signals.reference import reference_root

    history = history_root()
    assert history.name == "history"
    assert history.parent.name == "evals"
    assert history.parent.parent == reference_root().parent.parent
    assert (history.parent.parent / "pyproject.toml").exists()
