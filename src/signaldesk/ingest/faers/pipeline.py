"""Ingest one quarter, end to end.

The shape of the loop is dictated by disk: fetch a quarter, process it fully,
delete its raw files, then move to the next. Nothing accumulates. `staged_quarter`
owns the deletion so it happens even when this module raises.

Stage 1 deduplication runs here because both halves of it are quarter-local: the
deleted-cases list ships inside the archive, and version resolution within a
quarter is a straight group-by. Cross-quarter version resolution and the
probabilistic stage 2 need the whole corpus and live in `dedup.py`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from signaldesk.core.config import Settings, get_settings
from signaldesk.core.logging import get_logger
from signaldesk.ingest.faers import load, manifest
from signaldesk.ingest.faers.discover import Discovery, url_for
from signaldesk.ingest.faers.download import staged_quarter
from signaldesk.ingest.faers.parse import (
    NullCounts,
    attach_seriousness,
    parse_demo,
    parse_drug,
    parse_reac,
    parse_simple,
    parse_ther,
)
from signaldesk.ingest.faers.quarter import Quarter
from signaldesk.ingest.faers.reader import locate, read_deleted_cases, read_table
from signaldesk.ingest.faers.schemas import Table

log = get_logger(__name__)


@dataclass(slots=True)
class QuarterResult:
    """What one quarter's ingest produced."""

    quarter: Quarter
    row_counts: dict[str, int] = field(default_factory=dict)
    null_counts: dict[str, int] = field(default_factory=dict)
    date_precision: dict[str, int] = field(default_factory=dict)
    stage1_removed_versions: int = 0
    stage1_removed_deleted: int = 0
    had_deleted_file: bool = False
    prod_ai_present: int = 0
    prod_ai_total: int = 0
    seconds: float = 0.0
    bytes_downloaded: int = 0
    checksum: str = ""
    skipped: bool = False
    reason: str = ""


def stage1_within_quarter(
    cases: pl.DataFrame, deleted_cases: set[int]
) -> tuple[pl.DataFrame, int, int]:
    """Apply the published deduplication rule inside one quarter.

    Two parts, both of which the source defines rather than us:

    * A case identifier appearing more than once in a quarter keeps only its
      highest version. Earlier versions are superseded reports of the same case.
    * Case identifiers on the quarter's deleted list are withdrawn by the source
      and are removed outright.

    Returns the surviving cases and the counts removed by each rule. Version
    resolution *across* quarters is a corpus-wide operation and happens later.
    """
    before = cases.height
    survivors = cases.sort(
        ["caseid", "caseversion"], descending=[False, True], nulls_last=True
    ).unique(subset=["caseid"], keep="first", maintain_order=True)
    removed_versions = before - survivors.height

    if deleted_cases:
        kept = survivors.filter(~pl.col("caseid").is_in(list(deleted_cases)))
        removed_deleted = survivors.height - kept.height
        survivors = kept
    else:
        removed_deleted = 0

    return survivors, removed_versions, removed_deleted


def _restrict_children(frame: pl.DataFrame, keep: pl.Series) -> pl.DataFrame:
    """Drop child rows whose case did not survive stage 1."""
    return frame.filter(pl.col("primaryid").is_in(keep.implode()))


def process_extracted(
    root: Path,
    quarter: Quarter,
    result: QuarterResult,
    settings: Settings | None = None,
    *,
    to_postgres: bool = True,
) -> QuarterResult:
    """Parse, deduplicate and persist an already-extracted quarter.

    Split out from the download so the whole pipeline can be exercised against
    the committed fixture slice without touching the network.
    """
    settings = settings or get_settings()
    files = locate(root, quarter)
    result.had_deleted_file = files.has_deleted_file
    deleted = read_deleted_cases(files.deleted_cases, quarter)

    nulls = NullCounts()
    demo_raw = read_table(files.tables[Table.DEMO], Table.DEMO, quarter)
    cases, demo_nulls = parse_demo(demo_raw, quarter)
    nulls.merge(demo_nulls)

    outcomes, _ = parse_simple(read_table(files.tables[Table.OUTC], Table.OUTC, quarter), quarter)
    cases = attach_seriousness(cases, outcomes)
    cases = cases.with_columns(pl.lit(quarter.label).alias("quarter"))

    cases, removed_versions, removed_deleted = stage1_within_quarter(cases, deleted)
    result.stage1_removed_versions = removed_versions
    result.stage1_removed_deleted = removed_deleted
    keep = cases.get_column("primaryid")

    drugs, drug_nulls = parse_drug(
        read_table(files.tables[Table.DRUG], Table.DRUG, quarter), quarter
    )
    nulls.merge(drug_nulls)
    reactions, reac_nulls = parse_reac(
        read_table(files.tables[Table.REAC], Table.REAC, quarter), quarter
    )
    nulls.merge(reac_nulls)
    therapies, ther_nulls = parse_ther(
        read_table(files.tables[Table.THER], Table.THER, quarter), quarter
    )
    nulls.merge(ther_nulls)
    indications, indi_nulls = parse_simple(
        read_table(files.tables[Table.INDI], Table.INDI, quarter), quarter, "indi_pt"
    )
    nulls.merge(indi_nulls)
    sources, _ = parse_simple(read_table(files.tables[Table.RPSR], Table.RPSR, quarter), quarter)

    drugs = _restrict_children(drugs, keep)
    reactions = _restrict_children(reactions, keep)
    outcomes = _restrict_children(outcomes, keep)
    therapies = _restrict_children(therapies, keep).with_columns(pl.col("dur").alias("dur_days"))
    indications = _restrict_children(indications, keep)
    sources = _restrict_children(sources, keep)

    result.prod_ai_total = drugs.height
    result.prod_ai_present = int(drugs.get_column("prod_ai").is_not_null().sum())
    result.null_counts = nulls.as_dict()
    result.date_precision = {
        str(value): int(count)
        for value, count in cases.get_column("event_dt_precision").value_counts().rows()
    }

    datasets = [
        (load.CASE_TARGET, cases),
        (load.DRUG_TARGET, drugs),
        (load.REACTION_TARGET, reactions),
        (load.OUTCOME_TARGET, outcomes),
        (load.THERAPY_TARGET, therapies),
        (load.INDICATION_TARGET, indications),
        (load.RPSR_TARGET, sources),
    ]
    for target, frame in datasets:
        load.write_parquet(frame, target, quarter, settings)
        result.row_counts[target.dataset] = frame.height

    if to_postgres:
        for target, frame in datasets:
            load.copy_into(frame, target, quarter, settings)
    return result


def ingest_quarter(
    quarter: Quarter,
    discovery: Discovery,
    settings: Settings | None = None,
    *,
    force: bool = False,
    keep_raw: bool = False,
    to_postgres: bool = True,
) -> QuarterResult:
    """Download, parse, deduplicate within the quarter, and persist it."""
    settings = settings or get_settings()
    result = QuarterResult(quarter=quarter)

    decision = manifest.decide(quarter, checksum=None, force=force)
    if not decision.should_ingest:
        result.skipped = True
        result.reason = decision.reason
        log.info("faers.quarter.skipped", quarter=quarter.label, reason=decision.reason)
        return result

    started = time.monotonic()
    manifest.start(quarter)
    url = url_for(quarter, discovery)

    try:
        with staged_quarter(quarter, url, settings, keep_raw=keep_raw) as (root, archive):
            result.bytes_downloaded = archive.size_bytes
            result.checksum = archive.sha256
            process_extracted(root, quarter, result, settings, to_postgres=to_postgres)
        result.seconds = time.monotonic() - started
        manifest.complete(
            quarter,
            checksum=result.checksum,
            row_counts=result.row_counts,
            bytes_downloaded=result.bytes_downloaded,
            had_deleted_file=result.had_deleted_file,
        )
        log.info(
            "faers.quarter.ingested",
            quarter=quarter.label,
            cases=result.row_counts.get("case", 0),
            seconds=round(result.seconds, 1),
        )
    except Exception as exc:
        manifest.fail(quarter, f"{type(exc).__name__}: {exc}")
        raise

    return result
