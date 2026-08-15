"""BCPNN against PhViD, plus the properties the locked form has to keep."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from signaldesk.stats.bcpnn import information_component
from signaldesk.stats.types import Contingency

pytestmark = pytest.mark.unit


def _table(row: dict[str, Any]) -> Contingency:
    return Contingency(
        a=np.array([row["a"]], dtype=np.int64),
        b=np.array([row["b"]], dtype=np.int64),
        c=np.array([row["c"]], dtype=np.int64),
        d=np.array([row["d"]], dtype=np.int64),
    )


def test_posterior_mean_matches_phivid(reference_tables: list[dict[str, Any]]) -> None:
    """The derived posterior mean reproduces the published implementation.

    This is the test that validates the prior. If the pseudo-counts or the
    expression for gamma were wrong, the posterior mean would drift from PhViD's
    on the small-count tables while still looking plausible on the large ones.
    """
    for row in reference_tables:
        result = information_component(_table(row))
        assert float(result.ic_posterior_mean[0]) == pytest.approx(
            row["ic_posterior_mean"], rel=1e-9
        ), row["name"]


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


def test_locked_ic_is_the_observed_to_expected_ratio(
    reference_tables: list[dict[str, Any]],
) -> None:
    """The locked closed form, checked against its own definition.

    There is no published numeric example for pairing this closed form with the
    Bayesian variance - that combination is the ARCHITECTURE 8.2 lock, not a
    published estimator - so this asserts only that ``ic`` is log2 of the
    observed-to-expected ratio and nothing more. The prior, which is where a
    mistake would actually hide, is validated by the three tests above.
    """
    for row in reference_tables:
        table = _table(row)
        result = information_component(table)
        expected = np.log2(row["a"] / float(table.expected[0]))
        assert float(result.ic[0]) == pytest.approx(expected, rel=1e-12), row["name"]


def test_divergence_from_the_posterior_is_governed_by_the_expected_count(
    reference_tables: list[dict[str, Any]],
) -> None:
    """What separates the locked closed form from the posterior mean.

    Not the observed count. ``g`` scales as ``1/E``, so a pair with a small
    expected count carries a large prior and is shrunk hard, whatever ``a`` is:
    the ``a = 100, E = 0.4`` table diverges by 1.80 IC units while the
    ``a = 5000, E = 577`` table diverges by 0.002.

    This pins the behaviour so that a future change to the prior, which would
    move exactly this relationship, cannot pass unnoticed.
    """
    by_expected = sorted(
        (
            (
                float(_table(row).expected[0]),
                abs(
                    float(information_component(_table(row)).ic[0])
                    - float(information_component(_table(row)).ic_posterior_mean[0])
                ),
                row["name"],
            )
            for row in reference_tables
        ),
        reverse=True,
    )

    large_expected = [gap for expected, gap, _ in by_expected if expected >= 20.0]
    small_expected = [gap for expected, gap, _ in by_expected if expected < 1.0]

    assert large_expected, "fixture must exercise well-populated tables"
    assert small_expected, "fixture must exercise rare-pair tables"
    assert max(large_expected) < 0.1
    assert min(small_expected) > 1.0


def test_ic_is_undefined_at_zero_but_the_posterior_is_not() -> None:
    """``a = 0`` yields ``-inf`` for the locked form and a finite posterior.

    The contingency builder never emits such a row. The behaviour is pinned
    anyway so that a future change which starts emitting them fails here rather
    than writing negative infinity into the signal table.
    """
    table = Contingency(
        a=np.array([0], dtype=np.int64),
        b=np.array([500], dtype=np.int64),
        c=np.array([300], dtype=np.int64),
        d=np.array([99200], dtype=np.int64),
    )
    result = information_component(table)

    assert float(result.ic[0]) == -np.inf
    assert float(result.ic025[0]) == -np.inf
    assert np.isfinite(result.ic_posterior_mean[0])
    assert float(result.ic_posterior_mean[0]) < 0.0


def test_ic025_never_exceeds_ic(reference_tables: list[dict[str, Any]]) -> None:
    for row in reference_tables:
        result = information_component(_table(row))
        assert float(result.ic025[0]) <= float(result.ic[0])
        assert float(result.ic025_exact[0]) <= float(result.ic_posterior_mean[0])


def test_shrinkage_pulls_small_counts_toward_zero() -> None:
    """The reason BCPNN is in the panel at all.

    Two pairs with the same observed-to-expected ratio but a hundredfold
    difference in evidence must not get the same lower bound.
    """
    table = Contingency(
        a=np.array([3, 300], dtype=np.int64),
        b=np.array([97, 9700], dtype=np.int64),
        c=np.array([1000, 100000], dtype=np.int64),
        d=np.array([98900, 9890000], dtype=np.int64),
    )
    result = information_component(table)

    assert float(result.ic[0]) == pytest.approx(float(result.ic[1]), rel=1e-9)
    assert float(result.ic025[0]) < float(result.ic025[1])
