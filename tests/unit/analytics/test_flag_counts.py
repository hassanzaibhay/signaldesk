"""Headline flag counts exclude pairs below the minimum cell count.

Regression. ARCHITECTURE section 8.1 says pairs below ``a >= 3`` are computed
and flagged but excluded from headline counts. Only the ROR rule carries that
minimum in its own definition, so on the full corpus PRR alone flagged
4,039,013 pairs against 2,785,896 pairs with three or more cases: a headline
count of signals larger than the number of pairs eligible to be signals.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from signaldesk.analytics.signals import (
    _flag_counts,
    _ic_divergence,
    _ic_divergence_from_gap,
    _ic_gap,
    _merge_counts,
    read_run,
    signal_root,
)
from signaldesk.core.config import Settings
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
    for name in ("ror", "prr", "bcpnn", "ror_prr_bcpnn"):
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
    assert counts["ror_prr_bcpnn"] is not None


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


def test_the_three_way_conjunction_is_not_the_literal_any_three_of_four() -> None:
    """The old name read as a k-of-n rule. It never was one.

    Wherever MGPS flags a pair the other three do not, "any three of the four
    agreed" is a strictly larger set. On the corpus that gap is 19,322 pairs,
    which is exactly the size of the set MGPS adds.
    """
    table = Contingency(
        a=np.array([40, 40, 3], dtype=np.int64),
        b=np.array([60, 6_000, 20], dtype=np.int64),
        c=np.array([400, 400, 30], dtype=np.int64),
        d=np.array([9_999_500, 9_993_560, 9_999_947], dtype=np.int64),
    )
    panel = estimate_all(table, hyperparameters=PRIOR)

    conjunction = panel.flag_ror_prr_bcpnn
    stacked = np.vstack([panel.flag_ror, panel.flag_prr, panel.flag_bcpnn, panel.flag_mgps]).sum(
        axis=0
    )
    any_three = stacked >= 3

    assert conjunction.tolist() == (panel.flag_ror & panel.flag_prr & panel.flag_bcpnn).tolist()
    # The conjunction can never exceed the k-of-n set: it is a subset of it.
    assert int(np.sum(conjunction & ~any_three)) == 0


def test_bcpnn_nests_inside_ror_and_prr_on_the_corpus_shape() -> None:
    """Pinned from the corpus: IC025 > 0 is the binding constraint of the three.

    Measured over both 2026-08-15 runs, no BCPNN-flagged pair sits outside ROR
    or outside PRR, while 229,266 pairs satisfy ROR and PRR without satisfying
    BCPNN, which is why the conjunction count equals the BCPNN count exactly.
    That equality is real nesting rather than two columns aliasing each other,
    and this pins the direction so a change that made them genuinely identical
    would show up.
    """
    table = _table()
    panel = estimate_all(table, hyperparameters=PRIOR)

    strictly_bcpnn = panel.flag_bcpnn & ~(panel.flag_ror & panel.flag_prr)
    assert int(np.sum(strictly_bcpnn)) == 0
    assert panel.flag_ror_prr_bcpnn.tolist() == panel.flag_bcpnn.tolist()


def _wide_table() -> Contingency:
    """A table big enough to slice several ways."""
    rng = np.random.default_rng(20260816)
    a = rng.integers(0, 60, size=97).astype(np.int64)
    b = rng.integers(100, 5_000, size=97).astype(np.int64)
    c = rng.integers(100, 5_000, size=97).astype(np.int64)
    total = 10_000_000
    return Contingency(a=a, b=b, c=c, d=(total - a - b - c).astype(np.int64))


@pytest.mark.parametrize("chunk", [1, 7, 50, 96, 97, 1000])
def test_scoring_in_chunks_reproduces_scoring_the_table_whole(chunk: int) -> None:
    """The chunked path exists to fit the ingredient run into memory.

    It is only worth having if it changes nothing else. Every statistic here is
    per row, so a slice must score identically to the same rows inside the whole
    table - given the one cross-row quantity, the MGPS prior, fitted once and
    passed in.
    """
    table = _wide_table()
    whole = estimate_all(table, hyperparameters=PRIOR)

    counts: dict[str, int | None] = {}
    gaps = []
    ebgm05 = []
    for start in range(0, len(table), chunk):
        panel = estimate_all(
            table.rows(start, min(start + chunk, len(table))), hyperparameters=PRIOR
        )
        counts = _merge_counts(counts, _flag_counts(panel))
        gaps.append(_ic_gap(panel))
        ebgm05.append(panel.mgps.ebgm05)

    assert counts == _flag_counts(whole)
    assert _ic_divergence_from_gap(np.concatenate(gaps)) == _ic_divergence(whole)
    assert np.array_equal(np.concatenate(ebgm05), whole.mgps.ebgm05)


def test_a_null_count_survives_being_merged_across_chunks() -> None:
    """``all_four`` is null on a provisional prior. Summing would invent a count."""
    merged = _merge_counts({"all_four": None, "ror": 2}, {"all_four": None, "ror": 3})

    assert merged["all_four"] is None
    assert merged["ror"] == 5


def test_a_legacy_flag_column_is_renamed_when_the_run_is_read(
    tmp_path: Path, settings: Settings
) -> None:
    """Runs written before the rename are read through an alias, not rebuilt."""
    scoped = settings.model_copy(update={"data_dir": tmp_path / "data"})
    partition = signal_root(scoped) / "run=20260815T170508Z"
    partition.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"a": [3], "flag_three_of_four": [True]}).write_parquet(
        partition / "part-0.parquet"
    )

    frame = read_run("20260815T170508Z", scoped)

    assert "flag_ror_prr_bcpnn" in frame.columns
    assert "flag_three_of_four" not in frame.columns


def test_the_alias_is_a_no_op_once_a_run_carries_the_new_name(
    tmp_path: Path, settings: Settings
) -> None:
    """A run written after the rename must pass through untouched."""
    scoped = settings.model_copy(update={"data_dir": tmp_path / "data"})
    partition = signal_root(scoped) / "run=20260816T000000Z"
    partition.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"a": [3], "flag_ror_prr_bcpnn": [True]}).write_parquet(
        partition / "part-0.parquet"
    )

    frame = read_run("20260816T000000Z", scoped)

    assert frame.columns == ["a", "flag_ror_prr_bcpnn"]
