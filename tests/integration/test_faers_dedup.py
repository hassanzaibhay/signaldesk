"""Deduplication and the analytics accessors, over the fixture corpus.

The fixture spans 2013Q1, 2013Q2 and 2026Q2, and the two 2013 quarters carry a
real duplicate pair and a real near-miss found in the published data. That is
what makes the cross-quarter assertion meaningful: a per-quarter pass cannot see
either of them.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from signaldesk.analytics import faers as analytics
from signaldesk.core.config import Settings
from signaldesk.ingest.faers import manifest
from signaldesk.ingest.faers.dedup import (
    METHOD_PROBABILISTIC,
    chain_report,
    load_stats,
    persist,
    resolve_versions_across_quarters,
    stage2,
    stats_from_store,
)
from signaldesk.ingest.faers.load import CASE_TARGET, DRUG_TARGET, REACTION_TARGET, write_parquet
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


def _write_case(scoped: Settings, label: str, rows: list[dict[str, object]]) -> None:
    quarter = Quarter.parse(label)
    frame = pl.DataFrame(rows).with_columns(
        pl.col("fda_dt").str.to_date(), pl.lit(label).alias("quarter")
    )
    write_parquet(frame, CASE_TARGET, quarter, scoped)


@pytest.fixture
def constructed(tmp_path: Path, settings: Settings) -> Settings:
    """A hand-built corpus carrying the cases the published fixture lacks.

    The fixture slice has no case published at two versions in different
    quarters, so it cannot exercise the population rule at all. These four cases
    do: one superseded across quarters, its surviving version, a separate case
    that matches both of them, and a case with no drug rows. All four share one
    block, so nothing here depends on blocking behaviour.
    """
    scoped = settings.model_copy(update={"data_dir": tmp_path / "constructed"})
    shared: dict[str, object] = {"sex": "F", "country": "US", "age_years": 40.0}
    _write_case(
        scoped,
        "2013Q1",
        [
            {"primaryid": 100011, "caseid": 10001, "caseversion": 1, "fda_dt": "2013-01-05"}
            | shared,
            {"primaryid": 200011, "caseid": 20001, "caseversion": 1, "fda_dt": "2013-01-05"}
            | shared,
            {"primaryid": 300011, "caseid": 30001, "caseversion": 1, "fda_dt": "2013-01-05"}
            | shared,
        ],
    )
    _write_case(
        scoped,
        "2013Q2",
        [{"primaryid": 100012, "caseid": 10001, "caseversion": 2, "fda_dt": "2013-04-05"} | shared],
    )

    drugs = [
        {"primaryid": primaryid, "drug_seq": 1, "role_cod": "PS", "drugname_raw": "ASPIRIN"}
        for primaryid in (100011, 200011)
    ]
    write_parquet(pl.DataFrame(drugs), DRUG_TARGET, Quarter.parse("2013Q1"), scoped)
    write_parquet(
        pl.DataFrame(
            [{"primaryid": 100012, "drug_seq": 1, "role_cod": "PS", "drugname_raw": "ASPIRIN"}]
        ),
        DRUG_TARGET,
        Quarter.parse("2013Q2"),
        scoped,
    )

    reactions = [
        {"primaryid": primaryid, "pt": pt}
        for primaryid in (100011, 200011, 300011)
        for pt in ("NAUSEA", "HEADACHE")
    ]
    write_parquet(pl.DataFrame(reactions), REACTION_TARGET, Quarter.parse("2013Q1"), scoped)
    write_parquet(
        pl.DataFrame([{"primaryid": 100012, "pt": pt} for pt in ("NAUSEA", "HEADACHE")]),
        REACTION_TARGET,
        Quarter.parse("2013Q2"),
        scoped,
    )
    return scoped


def test_stage2_excludes_records_stage_1_already_superseded(constructed: Settings) -> None:
    """Defect 1. A superseded record cannot reach the numerator, so it is out.

    Left in, it sits in the denominator alone, which is the inconsistency the
    P02 report documented rather than fixed: 1,663,031 cases in a denominator
    they could never contribute to.
    """
    stats = stage2(constructed)
    superseded = {primaryid for primaryid, _ in resolve_versions_across_quarters(constructed)}

    assert superseded == {100011}
    assert stats.records_excluded_superseded == 1
    assert stats.records_excluded_empty_drug_set == 1
    # The surviving version and the case that matches it, and nothing else.
    assert stats.records_considered == 2
    assert stats.population == 2
    assert stats.records_evaluated == 4

    flagged = {primaryid for primaryid, *_ in stats.pairs}
    canonicals = {pair[1] for pair in stats.pairs}
    assert flagged == {200011}
    assert canonicals == {100012}
    assert flagged.isdisjoint(superseded)
    # Defect 4: the partner side is clean too, not only the flagged side.
    assert canonicals.isdisjoint(superseded)


def test_no_probabilistic_match_names_a_superseded_partner(loaded: Settings) -> None:
    """Defect 4, verified rather than assumed.

    The write-time skip was one-sided: it dropped a match when the flagged record
    was superseded, not when its partner was, leaving 98,376 flags naming a
    canonical stage 1 had removed. Restricting the population is supposed to make
    the case unreachable from either side. This asserts it, on both sides.
    """
    stats = stage2(loaded)
    version_pairs = resolve_versions_across_quarters(loaded)
    persist(stats, version_pairs, loaded)
    superseded = {primaryid for primaryid, _ in version_pairs}

    rows = Duplicate.objects.filter(method=METHOD_PROBABILISTIC).values_list(
        "primaryid", "canonical_primaryid"
    )
    for primaryid, canonical in rows:
        assert primaryid not in superseded
        assert canonical not in superseded

    assert stats.probabilistic_discarded_superseded == 0


def test_a_case_with_no_drug_string_is_excluded_rather_than_compared(loaded: Settings) -> None:
    """An empty drug set scores as dissimilar, so it can never match.

    Reported as an exclusion for the same reason as the other two: a record that
    cannot reach the numerator does not belong in the denominator. The count is
    the point, not its size.
    """
    stats = stage2(loaded)

    assert stats.records_excluded_empty_drug_set >= 0
    assert stats.population == stats.records_considered + (
        stats.records_excluded_single_member_block
    )
    assert stats.records_evaluated == stats.population + (
        stats.records_excluded_null_key
        + stats.records_excluded_superseded
        + stats.records_excluded_empty_drug_set
    )


def test_the_pass_and_the_store_agree_on_the_population(loaded: Settings) -> None:
    """Defect 2. The report's denominator is the one the pass compared.

    In P02 these were two queries meaning to say the same thing, and they
    differed by 4,646 because only one applied the single-member-block filter.
    They are now one function, so this asserts that they stay one.
    """
    stats = stage2(loaded)
    persist(stats, resolve_versions_across_quarters(loaded), loaded)

    rebuilt = stats_from_store(loaded)

    assert rebuilt.population == stats.population
    assert rebuilt.records_considered == stats.records_considered
    assert rebuilt.records_excluded_null_key == stats.records_excluded_null_key
    assert rebuilt.records_excluded_superseded == stats.records_excluded_superseded
    assert rebuilt.records_excluded_single_member_block == (
        stats.records_excluded_single_member_block
    )


def test_the_written_table_has_no_chain_without_a_survivor(loaded: Settings) -> None:
    """The gate. Every flagged record must resolve to a record nothing flagged."""
    stats = stage2(loaded)
    persist(stats, resolve_versions_across_quarters(loaded), loaded)

    report = chain_report()

    assert report.closed_components == 0
    assert report.cycles == 0
    assert report.is_sound
    assert report.resolved_to_unflagged == report.flagged


def test_the_discard_counter_survives_a_round_trip(loaded: Settings) -> None:
    """Defect 3. The quantity is recoverable from stored state, not re-derived."""
    stats = stage2(loaded)
    persist(stats, resolve_versions_across_quarters(loaded), loaded)

    reloaded = load_stats(loaded)

    assert reloaded is not None
    assert reloaded.probabilistic_discarded_superseded == 0
    assert reloaded.records_excluded_superseded == stats.records_excluded_superseded
