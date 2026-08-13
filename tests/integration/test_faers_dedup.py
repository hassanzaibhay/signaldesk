"""Deduplication and the analytics accessors, over the fixture corpus.

The fixture spans 2013Q1, 2013Q2 and 2026Q2, and the two 2013 quarters carry a
real duplicate pair and a real near-miss found in the published data. That is
what makes the cross-quarter assertion meaningful: a per-quarter pass cannot see
either of them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from signaldesk.analytics import faers as analytics
from signaldesk.core.config import Settings
from signaldesk.ingest.faers import manifest
from signaldesk.ingest.faers.dedup import (
    METHOD_PROBABILISTIC,
    persist,
    resolve_versions_across_quarters,
    stage2,
)
from signaldesk.ingest.faers.pipeline import QuarterResult, process_extracted
from signaldesk.ingest.faers.quarter import Quarter
from signaldesk.web.signals.models import Duplicate, IngestManifest

pytestmark = [pytest.mark.integration, pytest.mark.django_db(transaction=True)]

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "faers"
QUARTERS = ["2013Q1", "2013Q2", "2026Q2"]

#: The pair found in the published data: identical drug and reaction sets, same
#: sex, age and country, reported under different case identifiers a quarter
#: apart.
DUPLICATE_PAIR = (89871272, 92080091)

#: Reactions match exactly, drugs do not. Must never merge.
NEAR_MISS = (90394811, 93752461)


@pytest.fixture
def loaded(tmp_path: Path, settings: Settings) -> Settings:
    """Every fixture quarter loaded into Parquet and Postgres."""
    scoped = settings.model_copy(update={"data_dir": tmp_path / "data"})
    for label in QUARTERS:
        quarter = Quarter.parse(label)
        process_extracted(FIXTURES / label, quarter, QuarterResult(quarter=quarter), scoped)
    return scoped


def test_stage2_finds_the_real_cross_quarter_duplicate(loaded: Settings) -> None:
    stats = stage2(loaded)

    matched = {(pair[0], pair[1]) for pair in stats.pairs}
    flat = {primaryid for pair in stats.pairs for primaryid in (pair[0], pair[1])}

    assert DUPLICATE_PAIR[0] in flat or DUPLICATE_PAIR[1] in flat, matched
    assert stats.cross_quarter_pairs > 0
    assert stats.cross_quarter_share > 0


def test_the_near_miss_is_not_merged(loaded: Settings) -> None:
    stats = stage2(loaded)
    pairs = {(pair[0], pair[1]) for pair in stats.pairs}

    assert NEAR_MISS not in pairs
    assert tuple(reversed(NEAR_MISS)) not in pairs


def test_records_without_a_full_blocking_key_are_excluded(loaded: Settings) -> None:
    stats = stage2(loaded)

    assert stats.records_excluded_null_key > 0
    assert stats.excluded_share > 0
    # Excluded records are not silently counted as considered.
    assert stats.records_considered < stats.records_considered + stats.records_excluded_null_key


def test_filtering_beats_naive_comparison(loaded: Settings) -> None:
    stats = stage2(loaded)
    assert stats.comparisons <= stats.naive_comparisons


def test_persist_writes_stats_beside_its_own_data(loaded: Settings) -> None:
    """A scoped caller must not overwrite the production record.

    persist writes the pass measurements to disk. Defaulting that path meant a
    test run replaced the real corpus figures with fixture-sized ones, and the
    next report published them.
    """
    from signaldesk.ingest.faers.dedup import stats_path

    persist(stage2(loaded), resolve_versions_across_quarters(loaded), loaded)

    assert stats_path(loaded).is_file()
    assert loaded.data_dir in stats_path(loaded).parents


def test_persist_rebuilds_rather_than_appends(loaded: Settings) -> None:
    stats = stage2(loaded)
    versions = resolve_versions_across_quarters(loaded)

    first = persist(stats, versions, loaded)
    after_first = Duplicate.objects.count()
    second = persist(stats, versions, loaded)

    assert first == second
    assert Duplicate.objects.count() == after_first


def test_persisted_rows_carry_their_evidence(loaded: Settings) -> None:
    persist(stage2(loaded), resolve_versions_across_quarters(loaded), loaded)

    row = Duplicate.objects.filter(method=METHOD_PROBABILISTIC).first()
    assert row is not None
    assert row.drug_jaccard is not None
    assert row.reaction_jaccard is not None
    assert row.score == pytest.approx(min(row.drug_jaccard, row.reaction_jaccard))


def test_quarter_range_narrows_what_is_considered(loaded: Settings) -> None:
    everything = stage2(loaded)
    one_quarter = stage2(loaded, start=Quarter.parse("2013Q1"), end=Quarter.parse("2013Q1"))

    assert one_quarter.records_considered < everything.records_considered
    # Within a single quarter there is nothing cross-quarter to find.
    assert one_quarter.cross_quarter_pairs == 0


def test_analytics_accessors_read_the_parquet_store(loaded: Settings) -> None:
    counts = analytics.corpus_counts(loaded)

    assert counts.cases > 0
    assert counts.counts["drug"] > 0
    assert counts.counts["reaction"] > 0
    assert set(counts.counts) == set(analytics.DATASETS)


def test_analytics_rejects_an_unknown_dataset(loaded: Settings) -> None:
    with pytest.raises(ValueError, match="unknown dataset"):
        analytics.dataset_glob("not_a_dataset", loaded)


def test_prod_ai_coverage_is_zero_before_the_boundary(loaded: Settings) -> None:
    coverage = analytics.prod_ai_coverage(loaded).sort("quarter")
    by_quarter = dict(zip(coverage["quarter"], coverage["pct"], strict=True))

    assert by_quarter["2013Q1"] == 0.0
    assert by_quarter["2026Q2"] > 0.0


def test_case_level_accessors_are_bounded(loaded: Settings) -> None:
    drugs = analytics.drugs_for_case(DUPLICATE_PAIR[0], loaded)
    reactions = analytics.reactions_for_case(DUPLICATE_PAIR[0], loaded)

    assert drugs.height > 0
    assert reactions.height > 0
    assert set(drugs["primaryid"].to_list()) == {DUPLICATE_PAIR[0]}


def test_manifest_round_trip(loaded: Settings) -> None:
    quarter = Quarter.parse("2013Q1")

    assert manifest.decide(quarter, checksum=None).should_ingest is True

    manifest.start(quarter)
    manifest.complete(
        quarter,
        checksum="abc123",
        row_counts={"case": 120},
        bytes_downloaded=999,
        had_deleted_file=False,
    )

    assert manifest.decide(quarter, checksum="abc123").should_ingest is False
    assert manifest.decide(quarter, checksum="different").should_ingest is True
    assert manifest.decide(quarter, checksum="abc123", force=True).should_ingest is True
    assert manifest.completed_quarters() == [quarter]


def test_a_failed_unit_is_retried(loaded: Settings) -> None:
    quarter = Quarter.parse("2013Q2")
    manifest.start(quarter)
    manifest.fail(quarter, "boom")

    decision = manifest.decide(quarter, checksum=None)
    assert decision.should_ingest is True
    assert "failed" in decision.reason
    assert IngestManifest.objects.get(unit="2013Q2").error == "boom"
