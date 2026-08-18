"""The MGPS boundary diagnostic: is the bound the optimum, and does EBGM05 move?

The decision rule under test is one comparison - the likelihood at the ``alpha1``
lower bound against the best value anywhere on the grid. Monotonicity of the
curve is reported alongside but decides nothing, so a non-monotone profile whose
bound still wins must not come out as a defective fit, and a profile beaten by an
interior point must not come out as a genuine limit however tidy it looks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from signaldesk.analytics import mgps_diagnostic
from signaldesk.analytics.mgps_diagnostic import build_diagnostic, write_diagnostic
from signaldesk.analytics.signals import signal_root
from signaldesk.core.config import Settings
from signaldesk.stats.mgps import ProfilePoint

pytestmark = pytest.mark.unit

RUN = "20260815T170508Z"


def _write_run(settings: Settings, *, rows: int = 40) -> None:
    """A small scored run carrying the three columns the diagnostic reads."""
    rng = np.random.default_rng(20260816)
    a = rng.integers(1, 12, size=rows).astype(np.int64)
    partition = signal_root(settings) / f"run={RUN}"
    partition.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "a": a,
            "expected": (a * 0.6 + 0.4).astype(np.float64),
            "insufficient": a < 3,
        }
    ).write_parquet(partition / "part-0.parquet")


def _profile(values: dict[float, float]) -> tuple[ProfilePoint, ...]:
    """A canned profile: fixed alpha1 to negative log likelihood."""
    return tuple(
        ProfilePoint(
            value=value,
            neg_log_likelihood=nll,
            theta=(value, 0.5, 1.0, 1.0, 0.3),
            forward_nll=nll,
            forward_theta=(value, 0.5, 1.0, 1.0, 0.3),
            backward_nll=nll,
            backward_theta=(value, 0.5, 1.0, 1.0, 0.3),
        )
        for value, nll in values.items()
    )


@pytest.fixture
def scoped(tmp_path: Path, settings: Settings) -> Settings:
    scoped = settings.model_copy(update={"data_dir": tmp_path / "data"})
    _write_run(scoped)
    return scoped


def _stub_profile(monkeypatch: pytest.MonkeyPatch, values: dict[float, float]) -> None:
    monkeypatch.setattr(
        mgps_diagnostic,
        "profile_likelihood",
        lambda *args, **kwargs: _profile(values),
    )


def test_the_bound_winning_reads_as_a_genuine_optimum(
    scoped: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_profile(monkeypatch, {1e-6: 100.0, 1e-5: 101.0, 1e-3: 105.0, 0.1: 130.0})

    document = build_diagnostic(RUN, scoped, grid=(1e-5, 1e-3, 0.1), score_at=())

    assert document["bound_is_optimum"] is True
    assert document["nll_at_bound"] == 100.0
    assert document["grid_argmin"] == {"alpha1": 1e-5, "nll": 101.0}
    assert document["bound_minus_argmin"] == pytest.approx(-1.0)


def test_an_interior_point_beating_the_bound_reads_as_a_defective_fit(
    scoped: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bound is then an artefact of the search, not a limit of the data."""
    _stub_profile(monkeypatch, {1e-6: 120.0, 1e-5: 118.0, 1e-3: 100.0, 0.1: 130.0})

    document = build_diagnostic(RUN, scoped, grid=(1e-5, 1e-3, 0.1), score_at=())

    assert document["bound_is_optimum"] is False
    assert document["grid_argmin"] == {"alpha1": 1e-3, "nll": 100.0}
    assert document["bound_minus_argmin"] == pytest.approx(20.0)


def test_a_non_monotone_curve_does_not_by_itself_condemn_the_bound(
    scoped: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Monotonicity is an observation about shape, not the verdict.

    The corpus profile dips between two points far up the curve while the bound
    still wins. Reading that dip as a failure would withhold a correct result.
    """
    _stub_profile(monkeypatch, {1e-6: 100.0, 1e-5: 101.0, 1e-3: 140.0, 0.1: 130.0})

    document = build_diagnostic(RUN, scoped, grid=(1e-5, 1e-3, 0.1), score_at=())

    assert document["monotone_away_from_bound"] is False
    assert document["bound_is_optimum"] is True


def test_ebgm05_is_scored_at_the_bound_and_the_argmin_even_if_unrequested(
    scoped: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Those two points are what the decision rule turns on."""
    _stub_profile(monkeypatch, {1e-6: 100.0, 1e-5: 101.0, 1e-3: 105.0, 0.1: 130.0})

    document = build_diagnostic(RUN, scoped, grid=(1e-5, 1e-3, 0.1), score_at=())

    scored = [row["alpha1"] for row in document["ebgm05_sensitivity"]]
    assert scored == [1e-6, 1e-5]
    assert document["ebgm05_at_bound"]["alpha1"] == 1e-6
    assert document["ebgm05_at_grid_argmin"]["alpha1"] == 1e-5


def test_the_flagged_count_uses_the_sufficient_denominator(
    scoped: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run record's headline counts exclude insufficient pairs; so must this.

    Counting over every pair would give a number that reads as comparable to the
    run's MGPS flag count while being drawn from a different denominator.
    """
    _stub_profile(monkeypatch, {1e-6: 100.0, 1e-5: 101.0})

    document = build_diagnostic(RUN, scoped, grid=(1e-5,), score_at=())

    frame = pl.read_parquet(signal_root(scoped) / f"run={RUN}" / "part-0.parquet")
    sufficient = int((~frame["insufficient"]).sum())
    assert document["sufficient_pairs"] == sufficient
    assert document["pairs"] > sufficient
    for row in document["ebgm05_sensitivity"]:
        assert row["flagged_ebgm05_gt_2"] <= sufficient


def test_a_grid_entirely_at_or_below_the_bound_is_refused(scoped: Settings) -> None:
    """There is nothing to compare the bound against, so there is no verdict."""
    with pytest.raises(ValueError, match="above the lower bound"):
        build_diagnostic(RUN, scoped, grid=(1e-6, 1e-9))


def test_the_document_is_written_as_json(
    tmp_path: Path, scoped: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_profile(monkeypatch, {1e-6: 100.0, 1e-5: 101.0})
    out = tmp_path / "nested" / "diagnostic.json"

    returned = write_diagnostic(RUN, out, scoped, grid=(1e-5,), score_at=())
    written: dict[str, Any] = json.loads(out.read_text(encoding="utf-8"))

    assert written["bound_is_optimum"] == returned["bound_is_optimum"]
    assert written["run_id"] == RUN
