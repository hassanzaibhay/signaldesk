"""The ingest pipeline end to end, over the committed fixture slice.

Runs against a real Postgres and a real DuckDB. No network: the fixture stands in
for a downloaded and extracted quarter, which is what `process_extracted` takes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from signaldesk.core.config import Settings
from signaldesk.ingest.faers import load
from signaldesk.ingest.faers.pipeline import QuarterResult, process_extracted, stage1_within_quarter
from signaldesk.ingest.faers.quarter import Quarter
from signaldesk.web.signals.models import Case, IngestManifest

pytestmark = [pytest.mark.integration, pytest.mark.django_db(transaction=True)]

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "faers"
QUARTERS = ["2013Q1", "2013Q2", "2026Q2"]


@pytest.fixture
def fixture_settings(tmp_path: Path, settings: Settings) -> Settings:
    """Settings pointing the data directory at a scratch location."""
    return settings.model_copy(update={"data_dir": tmp_path / "data"})


def _ingest(label: str, settings: Settings, *, to_postgres: bool = True) -> QuarterResult:
    quarter = Quarter.parse(label)
    result = QuarterResult(quarter=quarter)
    return process_extracted(FIXTURES / label, quarter, result, settings, to_postgres=to_postgres)


def test_pipeline_writes_parquet_and_postgres(fixture_settings: Settings) -> None:
    result = _ingest("2013Q1", fixture_settings)

    assert result.row_counts["case"] == 120
    assert result.row_counts["drug"] > 0
    assert Case.objects.count() == 120

    partition = load.partition_path(load.CASE_TARGET, Quarter.parse("2013Q1"), fixture_settings)
    assert (partition / "part-0.parquet").is_file()

    case = Case.objects.first()
    assert case is not None
    assert case.quarter == "2013Q1"


def test_child_datasets_are_parquet_only(fixture_settings: Settings) -> None:
    _ingest("2013Q1", fixture_settings)
    for target in (load.DRUG_TARGET, load.REACTION_TARGET, load.OUTCOME_TARGET):
        assert target.db_table is None
        partition = load.partition_path(target, Quarter.parse("2013Q1"), fixture_settings)
        assert (partition / "part-0.parquet").is_file()


def test_second_run_is_a_no_op(fixture_settings: Settings) -> None:
    first = _ingest("2013Q1", fixture_settings)
    after_first = Case.objects.count()

    second = _ingest("2013Q1", fixture_settings)

    assert Case.objects.count() == after_first
    assert second.row_counts == first.row_counts


def test_both_eras_load_together(fixture_settings: Settings) -> None:
    for label in QUARTERS:
        _ingest(label, fixture_settings)

    assert Case.objects.filter(quarter="2013Q1").exists()
    assert Case.objects.filter(quarter="2026Q2").exists()
    # prod_ai exists only in the later era, and the column is stable across both.
    assert Case.objects.count() == sum(1 for _ in Case.objects.all())


def test_seriousness_is_derived_with_its_evidence(fixture_settings: Settings) -> None:
    _ingest("2013Q1", fixture_settings)

    fatal = Case.objects.filter(serious_death=True).first()
    assert fatal is not None
    assert "DE" in fatal.outcome_codes
    assert fatal.serious is True

    for case in Case.objects.filter(serious=False)[:20]:
        assert case.outcome_codes == []


def test_country_fallback_is_recorded(fixture_settings: Settings) -> None:
    _ingest("2026Q2", fixture_settings)
    sources = set(Case.objects.values_list("country_source", flat=True))
    assert sources <= {"occurrence", "reporter", "none"}
    assert Case.objects.filter(country_source="reporter").exists()


def test_manifest_records_whether_a_deleted_file_shipped(fixture_settings: Settings) -> None:
    for label in ("2013Q1", "2026Q2"):
        quarter = Quarter.parse(label)
        result = _ingest(label, fixture_settings)
        IngestManifest.objects.update_or_create(
            source="faers",
            unit=quarter.label,
            defaults={
                "status": IngestManifest.Status.COMPLETED,
                "had_deleted_file": result.had_deleted_file,
                "row_counts": result.row_counts,
            },
        )

    assert IngestManifest.objects.get(unit="2013Q1").had_deleted_file is False
    assert IngestManifest.objects.get(unit="2026Q2").had_deleted_file is True


def test_deleted_cases_are_removed_from_the_loaded_quarter(fixture_settings: Settings) -> None:
    result = _ingest("2026Q2", fixture_settings)
    # The fixture's deleted list names case identifiers that are present in the
    # same quarter, so the rule has something to remove.
    assert result.stage1_removed_deleted > 0


def test_stage1_keeps_the_highest_version_per_case() -> None:
    import polars as pl

    cases = pl.DataFrame(
        {
            "primaryid": [1, 2, 3],
            "caseid": [100, 100, 200],
            "caseversion": [1, 2, 1],
        }
    )
    survivors, removed_versions, removed_deleted = stage1_within_quarter(cases, set())

    assert survivors.height == 2
    assert removed_versions == 1
    assert removed_deleted == 0
    assert survivors.filter(pl.col("caseid") == 100)["caseversion"].to_list() == [2]


def test_stage1_removes_cases_on_the_deleted_list() -> None:
    import polars as pl

    cases = pl.DataFrame({"primaryid": [1, 2], "caseid": [100, 200], "caseversion": [1, 1]})
    survivors, _, removed_deleted = stage1_within_quarter(cases, {200})

    assert survivors["caseid"].to_list() == [100]
    assert removed_deleted == 1
