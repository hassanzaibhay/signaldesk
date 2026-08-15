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

from signaldesk.core.errors import IngestError
from signaldesk.core.logging import get_logger
from signaldesk.ingest.faers.dedup import ChainReport, DedupStats
from signaldesk.ingest.faers.pipeline import QuarterResult

log = get_logger(__name__)

HISTORY_DIR = Path("evals/history")


def build_report(
    results: list[QuarterResult],
    dedup: DedupStats | None,
    *,
    version_pairs: int = 0,
    corpus: dict[str, dict[str, int]] | None = None,
    postgres_cases: int | None = None,
    chains: ChainReport | None = None,
) -> dict[str, Any]:
    """Assemble the report from what the run actually measured.

    `corpus` carries figures counted from the stored Parquet rather than
    accumulated during ingest. When present they take precedence, because they
    describe the corpus as it exists rather than as one run happened to see it,
    and they survive a report rebuilt long after the ingest.

    `postgres_cases` is required whenever duplicate figures are reported. The
    flag counts come from `faers_duplicate` in Postgres, and an earlier version
    of this report divided them by the Parquet publication total, which is 707
    rows larger. Both terms of every rate now come from the same store, and each
    figure says which store that is.
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
        if postgres_cases is None:
            message = (
                "the duplicate rates need a Postgres case count: the flag counts "
                "come from faers_duplicate, and dividing them by the Parquet "
                "publication total mixes two stores across the gap ADR 0004 records"
            )
            raise IngestError(message)
        parquet_cases = (corpus or {}).get("case_field_nulls", {}).get("cases", cases)
        report["corpus_totals"] = {
            "cases_postgres": postgres_cases,
            "cases_parquet": parquet_cases,
            "gap": parquet_cases - postgres_cases,
            "note": (
                "Parquet holds one row per publication and Postgres one per case. "
                "Every rate below divides Postgres counts by Postgres counts; the "
                "Parquet total is reported for reconciliation only. See ADR 0004."
            ),
        }
        by_version = dedup.records_flagged_version
        by_probability = dedup.records_flagged_probabilistic
        report["rates"] = _rates(
            total_cases=postgres_cases,
            blockable=dedup.population,
            flagged_total=dedup.duplicate_records,
            flagged_by_version=by_version if by_version is not None else version_pairs,
            flagged_by_probability=by_probability,
        )
        report["deduplication"].update(
            {
                "stage2_source": "stored results" if dedup.from_store else "measured by this pass",
                "stage2_population_store": "duckdb over parquet, one row per case",
                "stage2_flag_store": "postgres faers_duplicate",
                "stage2_records_evaluated": dedup.records_evaluated,
                "stage2_population": dedup.population,
                "stage2_records_compared": dedup.records_considered,
                "stage2_records_alone_in_block": dedup.records_excluded_single_member_block,
                "stage2_records_excluded_null_blocking_key": dedup.records_excluded_null_key,
                "stage2_records_excluded_superseded": dedup.records_excluded_superseded,
                "stage2_records_excluded_empty_drug_set": dedup.records_excluded_empty_drug_set,
                "stage2_matches_discarded_superseded": dedup.probabilistic_discarded_superseded,
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
                "stage2_rate": _rate(dedup.duplicate_records, dedup.population),
                "interpretation": (
                    "The stage 2 rate is a lower bound. Three classes of case are "
                    "excluded because the rule cannot reach them and they would "
                    "otherwise sit in the denominator alone: records missing sex, age "
                    "or country, since a null key is not evidence of similarity; "
                    "records stage 1 already superseded; and records with no drug "
                    "string, which an empty-set comparison scores as dissimilar. A "
                    "case alone in its block is not excluded - the rule applied and "
                    "found nothing, which is a legitimate zero."
                ),
                "provisional": (
                    "Stage 2 compares raw drug name strings, so one ingredient under "
                    "several trade names counts as several set members. These figures "
                    "are provisional until the pass is repeated against normalized "
                    "ingredients, which is expected to raise the match rate."
                ),
            }
        )

    if chains is not None:
        report["chain_verification"] = {
            "store": "postgres faers_duplicate",
            "flagged_records": chains.flagged,
            "resolved_to_an_unflagged_record": chains.resolved_to_unflagged,
            "resolved_into_a_cycle": chains.resolved_into_cycle,
            "components": chains.components,
            "closed_components": chains.closed_components,
            "records_in_closed_components": chains.records_in_closed_components,
            "cycles": chains.cycles,
            "cycle_lengths": list(chains.cycle_lengths),
            "sound": chains.is_sound,
            "note": (
                "The surviving-case count subtracts flagged records from the corpus, "
                "which assumes every flagged record has a surviving representative. "
                "Both counts must be zero for that to hold; a closed component is a "
                "set of cases removed outright rather than merged into a survivor."
            ),
        }

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
    provisional = (
        "Provisional. Stage 2 is probabilistic and its false-positive rate has "
        "not been measured: 58 percent of its matches come from records carrying "
        "one drug and one reaction, where the similarity test reduces to exact "
        "agreement on two common values. Nothing here yet distinguishes a high "
        "true duplicate rate from a high false merge rate. Pending a hand audit "
        "of sampled pairs, this figure does not go in the README."
    )
    return {
        "stage1_superseded_of_all_cases": {
            "numerator": flagged_by_version,
            "denominator": total_cases,
            "population": "every case in the corpus",
            "store": "postgres",
            "rate": _rate(flagged_by_version, total_cases),
            "status": (
                "Settled. Version supersession is the published rule applied "
                "deterministically, not an estimate."
            ),
        },
        "stage2_matched_of_eligible_cases": {
            "numerator": probabilistic,
            "denominator": blockable,
            "population": (
                "cases the rule can apply to: stage 1 survivors with a complete "
                "blocking key and a non-empty drug set, including those alone in "
                "their block"
            ),
            "store": "postgres numerator, duckdb over parquet denominator, one row per case",
            "rate": _rate(probabilistic, blockable),
            "status": provisional,
        },
        "overall_duplicate_of_all_cases": {
            "numerator": flagged_total,
            "denominator": total_cases,
            "population": "every case in the corpus",
            "store": "postgres",
            "rate": _rate(flagged_total, total_cases),
            "status": provisional,
        },
        "unique_cases": {
            "count": max(total_cases - flagged_total, 0),
            "store": "postgres",
            "note": (
                "Cases minus flagged records, both counted in Postgres. Sound only "
                "while every flagged record's canonical chain ends at an unflagged "
                "record; see chain_verification."
            ),
        },
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
            "stage2_matched_of_eligible_cases",
            "overall_duplicate_of_all_cases",
        ):
            entry = rates[key]
            lines.append(
                f"  {key:<38} {entry['rate']:>8.2%}"
                f"  ({entry['numerator']:,} / {entry['denominator']:,}, {entry['store']})"
            )
        unique = rates["unique_cases"]
        lines.append(f"  {'unique cases':<38} {unique['count']:>8,}  ({unique['store']})")

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
        for key, label in [
            ("stage2_population", "population, rule applies"),
            ("stage2_records_compared", "of those, compared"),
            ("stage2_records_alone_in_block", "of those, alone in block"),
            ("stage2_records_excluded_null_blocking_key", "excluded, null blocking key"),
            ("stage2_records_excluded_superseded", "excluded, superseded by stage 1"),
            ("stage2_records_excluded_empty_drug_set", "excluded, empty drug set"),
        ]:
            lines.append(f"  {label:<34}{dedup[key]:>12,}")
        discarded = dedup.get("stage2_matches_discarded_superseded")
        lines.append(
            f"  {'matches discarded, superseded':<34}"
            + (f"{discarded:>12,}" if discarded is not None else "  not recorded")
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

    chains = report.get("chain_verification")
    if chains:
        lines.append("")
        lines.append("chain verification (postgres faers_duplicate)")
        lines.append(f"  {'closed components':<34}{chains['closed_components']:>12,}")
        lines.append(f"  {'cycles':<34}{chains['cycles']:>12,}")
        lines.append(
            f"  {'records with no survivor':<34}{chains['records_in_closed_components']:>12,}"
        )
        lines.append(f"  {'sound':<34}{chains['sound']!s:>12}")

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
