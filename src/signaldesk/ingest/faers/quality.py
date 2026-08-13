"""The ingest quality report.

Written to `evals/history/` and committed, because a number that only ever
existed in a terminal cannot be checked by anyone later. Every figure here is
counted during the run rather than estimated afterwards.

The duplicate rate is the headline, and it carries two qualifiers that must
travel with it: it is measured against the corpus after the published rule has
already removed superseded versions, and it excludes records too sparse to block.
Reporting the rate without the exclusion share would overstate how much of the
corpus was actually examined.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from signaldesk.core.logging import get_logger
from signaldesk.ingest.faers.dedup import DedupStats
from signaldesk.ingest.faers.pipeline import QuarterResult

log = get_logger(__name__)

HISTORY_DIR = Path("evals/history")


def build_report(
    results: list[QuarterResult],
    dedup: DedupStats | None,
    *,
    version_pairs: int = 0,
    corpus: dict[str, dict[str, int]] | None = None,
) -> dict[str, Any]:
    """Assemble the report from what the run actually measured.

    `corpus` carries figures counted from the stored Parquet rather than
    accumulated during ingest. When present they take precedence, because they
    describe the corpus as it exists rather than as one run happened to see it,
    and they survive a report rebuilt long after the ingest.
    """
    ingested = [result for result in results if not result.skipped]

    row_totals: dict[str, int] = {}
    null_totals: dict[str, int] = {}
    precision_totals: dict[str, int] = {}
    for result in ingested:
        for dataset, count in result.row_counts.items():
            row_totals[dataset] = row_totals.get(dataset, 0) + count
        for reason, count in result.null_counts.items():
            null_totals[reason] = null_totals.get(reason, 0) + count
        for precision, count in result.date_precision.items():
            precision_totals[precision] = precision_totals.get(precision, 0) + count

    cases = row_totals.get("case", 0)
    # A result rebuilt from the manifest carries row counts but not the
    # per-quarter normalization counters, which only exist while ingest runs.
    counters_present = any(result.date_precision or result.null_counts for result in ingested)
    stage1_versions = sum(result.stage1_removed_versions for result in ingested)
    stage1_deleted = sum(result.stage1_removed_deleted for result in ingested)
    prod_ai_present = sum(result.prod_ai_present for result in ingested)
    prod_ai_total = sum(result.prod_ai_total for result in ingested)

    report: dict[str, Any] = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "quarters": {
            "ingested": [result.quarter.label for result in ingested],
            "skipped": [result.quarter.label for result in results if result.skipped],
            "without_deleted_cases_file": [
                result.quarter.label for result in ingested if not result.had_deleted_file
            ],
        },
        "rows": {
            "per_quarter": {result.quarter.label: result.row_counts for result in ingested},
            "totals": row_totals,
        },
        "deduplication": {
            # None rather than 0 when the run that produced these results is not
            # the run writing the report: the manifest does not store them, and
            # reporting zero would assert something nobody measured.
            "stage1_superseded_versions_within_quarter": stage1_versions
            if counters_present
            else None,
            "stage1_withdrawn_by_source": stage1_deleted if counters_present else None,
            "stage1_superseded_versions_across_quarters": version_pairs,
        },
        "normalization_nulls": null_totals,
        "date_precision_event_dt": precision_totals,
        "prod_ai_coverage": {
            "rows_with_value": prod_ai_present,
            "rows_total": prod_ai_total,
            "share": _rate(prod_ai_present, prod_ai_total),
            "note": (
                "The source only publishes prod_ai from 2014Q3 onward; earlier rows "
                "are null by construction, not by data quality."
            ),
        },
        "timing": {
            "seconds_per_quarter": {
                result.quarter.label: round(result.seconds, 1) for result in ingested
            },
            "bytes_downloaded_per_quarter": {
                result.quarter.label: result.bytes_downloaded for result in ingested
            },
        },
    }

    if corpus is not None:
        # Counted from the stored corpus, so they hold for a report rebuilt at
        # any time rather than only for the process that did the ingest.
        report["date_precision_event_dt"] = corpus["date_precision_event_dt"]
        report["case_field_nulls"] = corpus["case_field_nulls"]
        coverage = corpus["prod_ai"]
        report["prod_ai_coverage"] = {
            "rows_with_value": coverage["rows_with_value"],
            "rows_total": coverage["rows_total"],
            "share": _rate(coverage["rows_with_value"], coverage["rows_total"]),
            "note": report["prod_ai_coverage"]["note"],
        }

    if dedup is not None:
        total_cases = (corpus or {}).get("case_field_nulls", {}).get("cases", cases)
        by_version = dedup.records_flagged_version
        by_probability = dedup.records_flagged_probabilistic
        report["rates"] = _rates(
            total_cases=total_cases,
            blockable=dedup.records_considered,
            flagged_total=dedup.duplicate_records,
            flagged_by_version=by_version if by_version is not None else version_pairs,
            flagged_by_probability=by_probability,
        )
        report["deduplication"].update(
            {
                "stage2_source": "stored results" if dedup.from_store else "measured by this pass",
                "stage2_records_considered": dedup.records_considered,
                "stage2_records_excluded_null_blocking_key": dedup.records_excluded_null_key,
                "stage2_excluded_share": round(dedup.excluded_share, 6),
                # Only a pass measures these. Rebuilt from stored results they
                # are unavailable, and null says so where zero would not.
                "stage2_blocks": None if dedup.from_store else dedup.blocks,
                "stage2_largest_block": None if dedup.from_store else dedup.largest_block,
                "stage2_comparisons": None if dedup.from_store else dedup.comparisons,
                "stage2_comparisons_if_naive": (
                    None if dedup.from_store else dedup.naive_comparisons
                ),
                "stage2_duplicate_pairs": None if dedup.from_store else dedup.duplicate_pairs,
                "stage2_duplicate_records": dedup.duplicate_records,
                "stage2_cross_quarter_pairs": dedup.cross_quarter_pairs,
                "stage2_cross_quarter_share": round(dedup.cross_quarter_share, 6),
                # Records, not pairs. One record pairs with many, so a rate
                # computed from pairs runs past 100 percent and means nothing.
                "stage2_rate": _rate(dedup.duplicate_records, dedup.records_considered),
                "interpretation": (
                    "The stage 2 rate is a lower bound. Records missing sex, age or "
                    "country are excluded from blocking entirely rather than matched "
                    "on the remaining keys, because a null key is not evidence of "
                    "similarity and relaxing it would manufacture false merges."
                ),
                "provisional": (
                    "Stage 2 compares raw drug name strings, so one ingredient under "
                    "several trade names counts as several set members. These figures "
                    "are provisional until the pass is repeated against normalized "
                    "ingredients, which is expected to raise the match rate."
                ),
            }
        )

    return report


def _rates(
    *,
    total_cases: int,
    blockable: int,
    flagged_total: int,
    flagged_by_version: int,
    flagged_by_probability: int | None,
) -> dict[str, Any]:
    """The duplicate rates, each against the population its rule applies to.

    The two stages do not share a denominator, and treating them as if they did
    overstates the result. Version supersession is checked on every case in the
    corpus. Probabilistic matching is only attempted on cases with a complete
    blocking key, which is a strict subset. Summing the two numerators and
    dividing by the smaller denominator inflates the rate, which is what an
    earlier version of this report did.
    """
    probabilistic = (
        flagged_by_probability
        if flagged_by_probability is not None
        else max(flagged_total - flagged_by_version, 0)
    )
    return {
        "stage1_superseded_of_all_cases": {
            "numerator": flagged_by_version,
            "denominator": total_cases,
            "population": "every case in the corpus",
            "rate": _rate(flagged_by_version, total_cases),
        },
        "stage2_matched_of_blockable_cases": {
            "numerator": probabilistic,
            "denominator": blockable,
            "population": "cases with a complete blocking key",
            "rate": _rate(probabilistic, blockable),
        },
        "overall_duplicate_of_all_cases": {
            "numerator": flagged_total,
            "denominator": total_cases,
            "population": "every case in the corpus",
            "rate": _rate(flagged_total, total_cases),
        },
        "unique_cases": max(total_cases - flagged_total, 0),
    }


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def write_report(report: dict[str, Any], directory: Path = HISTORY_DIR) -> Path:
    """Write the report under a timestamped name and return the path."""
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    path = directory / f"ingest_faers_{stamp}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log.info("faers.quality.written", path=str(path))
    return path


def render(report: dict[str, Any]) -> str:
    """A readable summary for the terminal."""
    lines: list[str] = []
    quarters = report["quarters"]["ingested"]
    lines.append(
        f"quarters ingested: {len(quarters)}"
        + (f" ({quarters[0]} to {quarters[-1]})" if quarters else "")
    )
    missing = report["quarters"]["without_deleted_cases_file"]
    if missing:
        lines.append(
            f"quarters shipping no deleted-cases file: {len(missing)} ({', '.join(missing)})"
        )

    lines.append("")
    lines.append("rows written")
    for dataset, count in sorted(report["rows"]["totals"].items()):
        lines.append(f"  {dataset:<14} {count:>12,}")

    rates = report.get("rates")
    if rates:
        lines.append("")
        lines.append("duplicate rates, each against the population its rule applies to")
        for key in (
            "stage1_superseded_of_all_cases",
            "stage2_matched_of_blockable_cases",
            "overall_duplicate_of_all_cases",
        ):
            entry = rates[key]
            lines.append(
                f"  {key:<38} {entry['rate']:>8.2%}"
                f"  ({entry['numerator']:,} / {entry['denominator']:,}, {entry['population']})"
            )
        lines.append(f"  {'unique cases':<38} {rates['unique_cases']:>8,}")

    dedup = report["deduplication"]
    lines.append("")
    lines.append("deduplication")
    within = dedup["stage1_superseded_versions_within_quarter"]
    lines.append(
        "  stage 1 superseded within quarter  "
        + (f"{within:>12,}" if within is not None else "  not recorded")
    )
    lines.append(
        f"  stage 1 superseded across quarters {dedup['stage1_superseded_versions_across_quarters']:>12,}"
    )
    withdrawn = dedup["stage1_withdrawn_by_source"]
    lines.append(
        "  stage 1 withdrawn by source        "
        + (f"{withdrawn:>12,}" if withdrawn is not None else "  not recorded")
    )
    if "stage2_duplicate_pairs" in dedup:
        pairs = dedup["stage2_duplicate_pairs"]
        lines.append(
            "  stage 2 duplicate pairs            "
            + (f"{pairs:>12,}" if pairs is not None else "  not recorded")
        )
        lines.append(
            f"  stage 2 records flagged            {dedup.get('stage2_duplicate_records', 0):>12,}"
        )
        lines.append(
            f"  stage 2 cross-quarter share        {dedup['stage2_cross_quarter_share']:>12.4%}"
        )
        lines.append(
            f"  excluded, null blocking key        {dedup['stage2_records_excluded_null_blocking_key']:>12,}"
        )
        lines.append(
            f"  excluded share                     {dedup['stage2_excluded_share']:>12.4%}"
        )
        lines.append(f"  figures from                       {dedup['stage2_source']:>12}")

        # Only a pass measures these; a rebuild says so rather than printing 0.
        for key, label in [
            ("stage2_comparisons", "comparisons made"),
            ("stage2_comparisons_if_naive", "comparisons if naive"),
            ("stage2_largest_block", "largest block"),
            ("stage2_blocks", "blocks"),
        ]:
            value = dedup.get(key)
            lines.append(
                f"  {label:<34}" + (f"{value:>12,}" if value is not None else "  not recorded")
            )

    coverage = report["prod_ai_coverage"]
    lines.append("")
    lines.append(
        f"prod_ai coverage: {coverage['share']:.2%} of {coverage['rows_total']:,} drug rows"
    )

    lines.append("")
    lines.append("event date precision")
    for precision, count in sorted(report["date_precision_event_dt"].items()):
        lines.append(f"  {precision:<10} {count:>12,}")

    nulls = report["normalization_nulls"]
    if nulls:
        lines.append("")
        lines.append("values nulled during normalization")
        for reason, count in sorted(nulls.items()):
            lines.append(f"  {reason:<36} {count:>12,}")

    return "\n".join(lines)
