"""The contingency builder over the committed fixture slice.

Integration rather than unit: it needs the Parquet store and the Postgres
duplicate table. The arithmetic it feeds is unit-tested against published
implementations in ``tests/unit/stats/``. What is checked here is that the
population, the two marginals and the background are all the *same* population -
the failure this layer is actually prone to, and one that leaves every cell
non-negative and every estimator returning a plausible number.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from signaldesk.analytics.contingency import (
    ContingencyError,
    ContingencySpec,
    DrugKey,
    PairTable,
    RoleFilter,
    build,
)
from signaldesk.core.config import Settings
from signaldesk.ingest.faers.dedup import persist, resolve_versions_across_quarters, stage2
from signaldesk.ingest.faers.pipeline import QuarterResult, process_extracted
from signaldesk.ingest.faers.quarter import Quarter

pytestmark = [pytest.mark.integration, pytest.mark.django_db(transaction=True)]

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "faers"
QUARTERS = ["2013Q1", "2013Q2", "2026Q2"]


@pytest.fixture
def corpus(tmp_path: Path, settings: Settings) -> Settings:
    """Every fixture quarter loaded, deduplicated, and the judgements persisted."""
    scoped = settings.model_copy(update={"data_dir": tmp_path / "data"})
    for label in QUARTERS:
        quarter = Quarter.parse(label)
        process_extracted(FIXTURES / label, quarter, QuarterResult(quarter=quarter), scoped)
    persist(stage2(scoped), resolve_versions_across_quarters(scoped), scoped)
    return scoped


def _raw(settings: Settings, **overrides: object) -> PairTable:
    spec = ContingencySpec(drug_key=DrugKey.RAW_STRING, min_a=1, **overrides)  # type: ignore[arg-type]
    return build(spec, settings)


def test_every_row_sums_to_the_background(corpus: Settings) -> None:
    """The invariant that catches a marginal computed over a different set.

    If the drug margin came from the full case set while the background came
    from the deduplicated one, every cell would still be non-negative and every
    estimator would still return a number. Only the sum gives it away.
    """
    pairs = _raw(corpus)

    assert len(pairs) > 0
    assert np.all(pairs.table.n == pairs.n_cases)


def test_no_cell_is_negative(corpus: Settings) -> None:
    pairs = _raw(corpus)

    for cell in (pairs.table.a, pairs.table.b, pairs.table.c, pairs.table.d):
        assert np.all(cell >= 0), "inclusion-exclusion must hold for d"


def test_cell_a_never_exceeds_either_marginal(corpus: Settings) -> None:
    pairs = _raw(corpus)

    assert np.all(pairs.table.n_drug >= pairs.table.a)
    assert np.all(pairs.table.n_event >= pairs.table.a)


def test_restricting_to_primary_suspects_cannot_add_pairs(corpus: Settings) -> None:
    """The sensitivity analysis is a subset of the primary one, by construction.

    The background must not move with it: narrowing which drug rows count does
    not narrow the population those rows were drawn from.
    """
    both = _raw(corpus, roles=RoleFilter.PS_SS)
    primary = _raw(corpus, roles=RoleFilter.PS)

    assert primary.pairs_observed <= both.pairs_observed
    assert primary.n_cases == both.n_cases


def test_the_minimum_count_counts_rather_than_filters(corpus: Settings) -> None:
    """Low-count pairs stay in the table; ``min_a`` only reports how many are not.

    Dropping them here would remove them from the denominator downstream and,
    separately, would starve the MGPS fit of the low-count mass its first
    mixture component exists to model.
    """
    low = build(ContingencySpec(drug_key=DrugKey.RAW_STRING, min_a=1), corpus)
    high = build(ContingencySpec(drug_key=DrugKey.RAW_STRING, min_a=3), corpus)

    assert low.pairs_observed == high.pairs_observed
    assert len(low) == len(high)
    assert low.n_cases == high.n_cases
    assert high.pairs_sufficient <= low.pairs_sufficient
    assert high.pairs_sufficient == int(np.count_nonzero(high.table.a >= 3))
    assert np.any(high.table.a < 3), "the slice must exercise pairs below the minimum"


def test_ingredient_mode_refuses_rather_than_falling_back(corpus: Settings) -> None:
    """Without the normalization tables it must name the fix, not silently degrade.

    Falling back to raw strings here would produce a complete set of plausible
    numbers computed over a different notion of what a drug is, with nothing on
    the output saying so.
    """
    with pytest.raises(ContingencyError, match="normalize drugs"):
        build(ContingencySpec(drug_key=DrugKey.INGREDIENT, min_a=1), corpus)


def test_duplicates_are_removed_from_the_background(corpus: Settings) -> None:
    from signaldesk.web.signals.models import Case, Duplicate

    pairs = _raw(corpus)
    flagged = Duplicate.objects.values("primaryid").distinct().count()

    assert flagged > 0, "the fixture must contain a real duplicate for this to mean anything"
    assert pairs.duplicates_removed == flagged
    assert pairs.n_cases == Case.objects.count() - flagged


def test_the_spec_travels_with_the_result(corpus: Settings) -> None:
    """No number leaves here without saying which analysis produced it."""
    pairs = _raw(corpus, roles=RoleFilter.PS)
    params = pairs.spec.as_params()

    assert params["drug_key"] == "raw_string"
    assert params["role_codes"] == "PS"
    assert params["min_a"] == 1


def test_the_signal_build_writes_scored_rows_and_its_own_provenance(corpus: Settings) -> None:
    """The whole entry point, over the fixture slice.

    Otherwise this path is only ever exercised by a seven-minute corpus run,
    which is not a place to discover that the record does not round-trip.
    """
    import polars as pl

    from signaldesk.analytics.signals import (
        build as build_signals,
    )
    from signaldesk.analytics.signals import (
        latest_run,
        read_record,
        read_run,
    )

    spec = ContingencySpec(drug_key=DrugKey.RAW_STRING, min_a=3)
    record, partition = build_signals(spec, corpus)

    assert partition.exists()
    assert latest_run(corpus) == record.run_id

    frame = read_run(record.run_id, corpus)
    assert isinstance(frame, pl.DataFrame)
    assert frame.height == record.pairs_observed
    assert set(frame["run_id"].unique().to_list()) == {record.run_id}

    for column in ("a", "b", "c", "d", "ror", "prr", "chi2", "ic", "ic025", "ebgm05"):
        assert column in frame.columns

    # Every row's cells sum to the background, all the way through the build.
    totals = (
        frame.select(pl.col("a") + pl.col("b") + pl.col("c") + pl.col("d")).to_series().unique()
    )
    assert totals.to_list() == [record.n_cases]

    stored = read_record(record.run_id, corpus)
    assert stored["run_id"] == record.run_id
    assert stored["population"]["deduplicated_cases"] == record.n_cases
    assert stored["params"]["drug_key"] == "raw_string"
    assert "provisional" in stored["mgps"]


def test_a_second_build_does_not_overwrite_the_first(corpus: Settings) -> None:
    """Runs are immutable. A quoted number must stay traceable to its own run."""
    from signaldesk.analytics.signals import build as build_signals
    from signaldesk.analytics.signals import signal_root

    spec = ContingencySpec(drug_key=DrugKey.RAW_STRING, min_a=3)
    first, _ = build_signals(spec, corpus)
    second, _ = build_signals(
        ContingencySpec(drug_key=DrugKey.RAW_STRING, roles=RoleFilter.PS, min_a=3), corpus
    )

    runs = {path.name for path in signal_root(corpus).iterdir() if path.is_dir()}
    assert f"run={first.run_id}" in runs
    if first.run_id != second.run_id:
        assert f"run={second.run_id}" in runs
