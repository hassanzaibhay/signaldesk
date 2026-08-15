"""BCPNN against PhViD, and the divergence the ARCHITECTURE amendment resolved.

``ic`` is the posterior mean under the Bate (1998) prior.
``ic_observed_expected`` is the unshrunk closed form section 8.2 previously
specified, kept as a secondary column.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from signaldesk.stats.bcpnn import EXACT_SIGMA, IC025_SIGMA, information_component
from signaldesk.stats.types import Contingency

pytestmark = pytest.mark.unit


def _table(row: dict[str, Any]) -> Contingency:
    return Contingency(
        a=np.array([row["a"]], dtype=np.int64),
        b=np.array([row["b"]], dtype=np.int64),
        c=np.array([row["c"]], dtype=np.int64),
        d=np.array([row["d"]], dtype=np.int64),
    )


def test_reported_ic_is_the_posterior_mean_and_matches_phivid(
    reference_tables: list[dict[str, Any]],
) -> None:
    """The test that validates the prior.

    If the pseudo-counts or the expression for gamma were wrong, the posterior
    mean would drift from PhViD's on the small-count tables while still looking
    plausible on the large ones.
    """
    for row in reference_tables:
        result = information_component(_table(row))
        assert float(result.ic[0]) == pytest.approx(row["ic_posterior_mean"], rel=1e-9), row["name"]


def test_posterior_variance_matches_phivid(reference_tables: list[dict[str, Any]]) -> None:
    for row in reference_tables:
        result = information_component(_table(row))
        assert float(result.ic_variance[0]) == pytest.approx(row["ic_variance"], rel=1e-9), row[
            "name"
        ]


def test_exact_lower_bound_matches_phivid(reference_tables: list[dict[str, Any]]) -> None:
    for row in reference_tables:
        result = information_component(_table(row))
        assert float(result.ic025_exact[0]) == pytest.approx(row["ic_lower_2p5"], rel=1e-9), row[
            "name"
        ]


def test_the_secondary_column_is_the_observed_to_expected_ratio(
    reference_tables: list[dict[str, Any]],
) -> None:
    """``ic_observed_expected`` is exactly ``log2(a / E)`` and nothing more.

    There is no published numeric example to check it against, because on its
    own it is arithmetic with no free parameters. The prior, which is where a
    mistake would actually hide, is validated by the three tests above.
    """
    for row in reference_tables:
        table = _table(row)
        result = information_component(table)
        expected = np.log2(row["a"] / float(table.expected[0]))
        assert float(result.ic_observed_expected[0]) == pytest.approx(expected, rel=1e-12), row[
            "name"
        ]


def test_the_two_bounds_differ_only_by_the_sigma_multiplier(
    reference_tables: list[dict[str, Any]],
) -> None:
    """Both bounds now start from the posterior mean, so nothing else separates them.

    Before the amendment ``ic025`` started from the unshrunk closed form and the
    two bounds diverged by up to 11.72 IC units. This is the assertion that the
    amendment actually landed in the arithmetic and not only in the prose.
    """
    for row in reference_tables:
        result = information_component(_table(row))
        standard_deviation = float(np.sqrt(result.ic_variance[0]))
        gap = float(result.ic025_exact[0]) - float(result.ic025[0])

        assert gap == pytest.approx((IC025_SIGMA - EXACT_SIGMA) * standard_deviation, rel=1e-9)
        assert gap < 0.05, row["name"]


def test_the_unshrunk_column_diverges_as_the_expected_count_falls(
    reference_tables: list[dict[str, Any]],
) -> None:
    """Why the amendment was needed, pinned as a measurement.

    ``g`` scales as ``1/E``, so a pair between a rare drug and a rare term
    carries an enormous prior and its posterior mean is shrunk hard, while the
    closed form is not shrunk at all. The observed count is not what drives it:
    the ``a = 100, E = 0.4`` table diverges by 1.80 while ``a = 5000, E = 577``
    diverges by 0.002.
    """
    measured = [
        (
            float(_table(row).expected[0]),
            abs(
                float(information_component(_table(row)).ic_observed_expected[0])
                - float(information_component(_table(row)).ic[0])
            ),
        )
        for row in reference_tables
    ]

    large_expected = [gap for expected, gap in measured if expected >= 20.0]
    small_expected = [gap for expected, gap in measured if expected < 1.0]

    assert large_expected, "fixture must exercise well-populated tables"
    assert small_expected, "fixture must exercise rare-pair tables"
    assert max(large_expected) < 0.1
    assert min(small_expected) > 1.0
    assert max(small_expected) > 10.0


def test_the_rare_pair_row_reconciles_to_the_published_numbers() -> None:
    """The worked row quoted in the ARCHITECTURE 8.2 amendment.

    Pinned because two different gaps come off this one row - 11.7566 between the
    point estimates and 11.7221 between the bounds - and the 0.0345 between them
    is the ``2`` against ``1.96`` multiplier, not a discrepancy.
    """
    table = Contingency(
        a=np.array([4], dtype=np.int64),
        b=np.array([6], dtype=np.int64),
        c=np.array([20], dtype=np.int64),
        d=np.array([999_970], dtype=np.int64),
    )
    result = information_component(table)
    standard_deviation = float(np.sqrt(result.ic_variance[0]))

    assert float(table.expected[0]) == pytest.approx(0.00024, rel=1e-3)
    assert float(result.ic_observed_expected[0]) == pytest.approx(14.0247, abs=1e-4)
    assert float(result.ic[0]) == pytest.approx(2.2681, abs=1e-4)
    assert standard_deviation == pytest.approx(0.8624, abs=1e-4)

    point_gap = float(result.ic_observed_expected[0]) - float(result.ic[0])
    bound_gap = (float(result.ic_observed_expected[0]) - IC025_SIGMA * standard_deviation) - float(
        result.ic025_exact[0]
    )

    assert point_gap == pytest.approx(11.7566, abs=1e-4)
    assert bound_gap == pytest.approx(11.7221, abs=1e-4)
    assert point_gap - bound_gap == pytest.approx(
        (IC025_SIGMA - EXACT_SIGMA) * standard_deviation, rel=1e-9
    )

    # What the amendment actually reports for this row.
    assert float(result.ic025[0]) == pytest.approx(0.5433, abs=1e-4)


def test_the_secondary_column_is_undefined_at_zero_but_the_reported_one_is_not() -> None:
    """``a = 0`` yields ``-inf`` for the ratio and a finite posterior mean.

    The contingency builder never emits such a row. The behaviour is pinned
    anyway so that a future change which starts emitting them fails here rather
    than writing negative infinity into the reported column.
    """
    table = Contingency(
        a=np.array([0], dtype=np.int64),
        b=np.array([500], dtype=np.int64),
        c=np.array([300], dtype=np.int64),
        d=np.array([99_200], dtype=np.int64),
    )
    result = information_component(table)

    assert float(result.ic_observed_expected[0]) == -np.inf
    assert np.isfinite(result.ic[0])
    assert np.isfinite(result.ic025[0])
    assert float(result.ic[0]) < 0.0
    assert not bool(result.ic025[0] > 0.0), "an unobserved pair must never flag"


def test_ic025_never_exceeds_ic(reference_tables: list[dict[str, Any]]) -> None:
    for row in reference_tables:
        result = information_component(_table(row))
        assert float(result.ic025[0]) <= float(result.ic[0])
        assert float(result.ic025_exact[0]) <= float(result.ic[0])


def test_shrinkage_pulls_small_counts_toward_zero() -> None:
    """The reason BCPNN is in the panel at all.

    Two pairs with the same observed-to-expected ratio but a hundredfold
    difference in evidence must not get the same answer. Under the amended
    definition they no longer even get the same point estimate, which is the
    behaviour the unshrunk closed form was hiding.
    """
    table = Contingency(
        a=np.array([3, 300], dtype=np.int64),
        b=np.array([97, 9_700], dtype=np.int64),
        c=np.array([1_000, 100_000], dtype=np.int64),
        d=np.array([98_900, 9_890_000], dtype=np.int64),
    )
    result = information_component(table)

    assert float(result.ic_observed_expected[0]) == pytest.approx(
        float(result.ic_observed_expected[1]), rel=1e-9
    )
    assert float(result.ic[0]) < float(result.ic[1])
    assert float(result.ic025[0]) < float(result.ic025[1])
