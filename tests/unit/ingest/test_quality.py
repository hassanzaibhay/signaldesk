"""The quality report: what it counts, and what it refuses to leave out."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from signaldesk.ingest.faers.dedup import DedupStats
from signaldesk.ingest.faers.pipeline import QuarterResult
from signaldesk.ingest.faers.quality import build_report, render, write_report
from signaldesk.ingest.faers.quarter import Quarter

pytestmark = pytest.mark.unit


def _result(label: str, **overrides: object) -> QuarterResult:
    result = QuarterResult(quarter=Quarter.parse(label))
    result.row_counts = {"case": 100, "drug": 400, "reaction": 300}
    result.null_counts = {"age_missing_or_unconvertible": 7}
    result.date_precision = {"day": 60, "month": 20, "year": 10, "missing": 10}
    result.stage1_removed_versions = 5
    result.stage1_removed_deleted = 2
    result.prod_ai_present = 0
    result.prod_ai_total = 400
    result.seconds = 12.5
    result.bytes_downloaded = 1000
    result.had_deleted_file = False
    for name, value in overrides.items():
        setattr(result, name, value)
    return result


def test_totals_sum_across_quarters() -> None:
    report = build_report([_result("2013Q1"), _result("2013Q2")], None)
    assert report["rows"]["totals"]["case"] == 200
    assert report["normalization_nulls"]["age_missing_or_unconvertible"] == 14
    assert report["date_precision_event_dt"]["day"] == 120


def test_skipped_quarters_are_listed_but_not_counted() -> None:
    skipped = _result("2013Q3", skipped=True)
    report = build_report([_result("2013Q1"), skipped], None)
    assert report["quarters"]["ingested"] == ["2013Q1"]
    assert report["quarters"]["skipped"] == ["2013Q3"]
    assert report["rows"]["totals"]["case"] == 100


def test_quarters_without_a_deleted_file_are_named() -> None:
    report = build_report([_result("2013Q1"), _result("2026Q2", had_deleted_file=True)], None)
    assert report["quarters"]["without_deleted_cases_file"] == ["2013Q1"]


def test_prod_ai_coverage_reports_its_own_caveat() -> None:
    report = build_report(
        [_result("2013Q1"), _result("2026Q2", prod_ai_present=398, prod_ai_total=400)], None
    )
    coverage = report["prod_ai_coverage"]
    assert coverage["rows_with_value"] == 398
    assert coverage["share"] == pytest.approx(398 / 800)
    assert "2014Q3" in coverage["note"]


def test_dedup_section_carries_the_exclusion_and_its_interpretation() -> None:
    stats = DedupStats(
        records_considered=600,
        records_excluded_null_key=400,
        blocks=12,
        largest_block=90,
        comparisons=1_000,
        naive_comparisons=180_000,
        duplicate_pairs=50,
        cross_quarter_pairs=35,
    )
    report = build_report([_result("2013Q1")], stats, version_pairs=9)
    dedup = report["deduplication"]

    assert dedup["stage2_records_excluded_null_blocking_key"] == 400
    assert dedup["stage2_excluded_share"] == pytest.approx(0.4)
    assert dedup["stage2_cross_quarter_share"] == pytest.approx(0.7)
    assert dedup["stage2_comparisons_if_naive"] == 180_000
    assert dedup["stage1_superseded_versions_across_quarters"] == 9
    # The caveat travels with the number, not in a commit message.
    assert "lower bound" in dedup["interpretation"]


def test_duplicate_rate_is_per_record_not_per_pair() -> None:
    """A rate over pairs exceeds 100 percent and means nothing.

    One record pairs with many others, so the full corpus produced 27 million
    pairs against 11.5 million records considered. Dividing those gives 235
    percent, which is how a meaningless number reaches a published artifact.
    """
    stats = DedupStats(records_considered=1000, duplicate_pairs=2500)
    stats.pairs = [(i, i + 1, 1.0, 1.0, False) for i in range(200)]

    report = build_report([_result("2013Q1")], stats)
    dedup = report["deduplication"]

    assert dedup["stage2_duplicate_records"] == 200
    assert dedup["stage2_rate"] == pytest.approx(0.2)
    assert dedup["stage2_rate"] <= 1.0


def test_counts_the_manifest_cannot_supply_read_as_unknown() -> None:
    """Zero would assert something nobody measured."""
    rebuilt = QuarterResult(quarter=Quarter.parse("2013Q1"))
    rebuilt.row_counts = {"case": 100}

    report = build_report([rebuilt], None)
    dedup = report["deduplication"]

    assert dedup["stage1_superseded_versions_within_quarter"] is None
    assert dedup["stage1_withdrawn_by_source"] is None
    assert "not recorded" in render(report)


def test_report_without_dedup_omits_the_stage2_keys() -> None:
    report = build_report([_result("2013Q1")], None)
    assert "stage2_duplicate_pairs" not in report["deduplication"]


def test_rendered_report_states_the_headline_numbers() -> None:
    stats = DedupStats(
        records_considered=600,
        records_excluded_null_key=400,
        duplicate_pairs=50,
        cross_quarter_pairs=35,
        blocks=12,
        largest_block=90,
        comparisons=1000,
        naive_comparisons=180_000,
    )
    text = render(build_report([_result("2013Q1")], stats))
    assert "quarters ingested: 1" in text
    assert "stage 2 duplicate pairs" in text
    assert "excluded, null blocking key" in text
    assert "comparisons if naive" in text


def test_report_is_written_as_sorted_json(tmp_path: Path) -> None:
    report = build_report([_result("2013Q1")], None)
    path = write_report(report, tmp_path)

    assert path.parent == tmp_path
    assert path.name.startswith("ingest_faers_")
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["rows"]["totals"]["case"] == 100
    assert list(written) == sorted(written)


def test_empty_run_does_not_divide_by_zero() -> None:
    report = build_report([], None)
    assert report["rows"]["totals"] == {}
    assert report["prod_ai_coverage"]["share"] == 0.0
    assert report["deduplication"]["stage1_rate"] == 0.0
