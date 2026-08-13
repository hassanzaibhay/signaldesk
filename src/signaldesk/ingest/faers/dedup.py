"""Corpus-wide deduplication.

Two things happen here that cannot happen inside a single quarter's ingest.

**Cross-quarter version resolution.** A case first reported in one quarter and
updated in a later one appears in both, at different versions. The published rule
keeps the highest version per case identifier, and that is only computable once
the quarters are all loaded.

**Probabilistic matching.** The same underlying event is routinely re-reported as
a *different* case identifier by a different reporter. Between 2013Q1 and 2013Q2
alone there are 429 such pairs. Blocking within one quarter cannot see any of
them, and a per-quarter duplicate rate would therefore be several times below
the truth while looking entirely plausible.

Both read Parquet through DuckDB rather than Postgres. This is a full-corpus
grouped comparison, which is what the analytical store is for.

## Why the comparison is not all-pairs

Blocking on (sex, rounded age, country) over two quarters gives 8,928 blocks with
a largest block of 1,918 records and 65.7 million within-block pairs. That scales
quadratically with corpus size, so at 55 quarters the largest blocks would be
tens of thousands of records and one block alone would exceed a billion pairs.

Inside a block we therefore use the standard set-similarity join filters rather
than comparing everything:

* **Size filter.** Jaccard(x, y) >= t implies |x| * t <= |y| <= |x| / t. Sets too
  different in size cannot reach the threshold and are never compared.
* **Prefix filter.** Order tokens by global rarity. For a set of size n and
  threshold t, two sets can only reach the threshold if their prefixes of length
  n - ceil(t * n) + 1 share a token. Candidates are generated from an inverted
  index over prefix tokens instead of from the cross product.

Both are exact: they discard only pairs that provably cannot meet the threshold,
so the result is identical to the naive comparison. The reported comparison count
is the number of pairs actually scored, next to what the naive count would have
been.

## What this deliberately does not do

A record missing sex, age, or country is **excluded from stage 2 entirely**,
rather than being placed in a null block or matched on the remaining keys. A null
blocking key is not evidence of similarity, and relaxing the key to include such
records would manufacture merges between cases whose only commonality is absent
data. The excluded count and share are reported.

The consequence, stated plainly because it bounds every number downstream: **the
reported duplicate rate is a lower bound.** Records too sparse to block reliably
cannot be matched, and the honest response is to report how many there were, not
to lower the bar until they match something.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import duckdb

from signaldesk.core.config import Settings, get_settings
from signaldesk.core.db import connect
from signaldesk.core.logging import get_logger
from signaldesk.ingest.faers.load import parquet_root
from signaldesk.ingest.faers.quarter import Quarter

log = get_logger(__name__)

DRUG_THRESHOLD = 0.8
REACTION_THRESHOLD = 0.8
MAX_AGE_DIFFERENCE_YEARS = 1.0

METHOD_VERSION = "max_caseversion"
METHOD_PROBABILISTIC = "probabilistic_v1"

#: Blocks larger than this are split by the prefix filter rather than compared
#: directly. The value is not a quality knob: the filters are exact, so this only
#: decides when the index is worth building.
BLOCK_INDEX_THRESHOLD = 50


@dataclass(slots=True)
class DedupStats:
    """Everything the quality report needs to describe a dedup run."""

    records_considered: int = 0
    records_excluded_null_key: int = 0
    blocks: int = 0
    largest_block: int = 0
    comparisons: int = 0
    naive_comparisons: int = 0
    duplicate_pairs: int = 0
    cross_quarter_pairs: int = 0
    superseded_versions: int = 0
    pairs: list[tuple[int, int, float, float, bool]] = field(default_factory=list)

    @property
    def cross_quarter_share(self) -> float:
        """Share of detected duplicates whose members sit in different quarters.

        The number that justifies running this corpus-wide: anything above zero
        is invisible to a per-quarter pass.
        """
        if not self.duplicate_pairs:
            return 0.0
        return self.cross_quarter_pairs / self.duplicate_pairs

    @property
    def excluded_share(self) -> float:
        total = self.records_considered + self.records_excluded_null_key
        return self.records_excluded_null_key / total if total else 0.0


@dataclass(frozen=True, slots=True)
class BlockRecord:
    """One case reduced to what deduplication needs to compare it.

    A typed record rather than a row mapping: every field here is read in the
    hot loop, and getting one of them wrong would silently change what counts as
    a duplicate.
    """

    index: int
    primaryid: int
    caseid: int
    quarter: str
    sex: str
    country: str
    age: float
    fda_dt: str
    drugs: frozenset[str]
    reactions: frozenset[str]

    @property
    def block_key(self) -> tuple[str, float, str]:
        return (self.sex, self.age, self.country)


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    """Set overlap. Two empty sets are not similar, they are uninformative."""
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    return intersection / (len(left) + len(right) - intersection)


def is_duplicate(
    drug_similarity: float,
    reaction_similarity: float,
    same_sex: bool,
    age_difference: float | None,
    same_country: bool,
) -> bool:
    """The stage 2 decision, isolated so it can be tested on hand-built cases.

    Every condition must hold. A pair matching on reactions but not on drugs is
    two patients with the same common adverse event, not one case reported twice.
    """
    if not (same_sex and same_country):
        return False
    if age_difference is None or age_difference > MAX_AGE_DIFFERENCE_YEARS:
        return False
    return drug_similarity >= DRUG_THRESHOLD and reaction_similarity >= REACTION_THRESHOLD


def prefix_length(size: int, threshold: float) -> int:
    """Tokens that must be indexed for a set to be findable at this threshold."""
    return max(1, size - math.ceil(threshold * size) + 1)


def _candidate_pairs(members: list[BlockRecord], threshold: float) -> set[tuple[int, int]]:
    """Pairs that could reach the threshold, by size and prefix filtering."""
    ordered = sorted(members, key=lambda record: len(record.reactions))
    index: dict[str, list[int]] = defaultdict(list)
    candidates: set[tuple[int, int]] = set()

    for position, record in enumerate(ordered):
        size = len(record.reactions)
        if size == 0:
            continue
        lower_bound = size * threshold
        prefix = sorted(record.reactions)[: prefix_length(size, threshold)]
        for token in prefix:
            for other in index[token]:
                if len(ordered[other].reactions) >= lower_bound:
                    candidates.add((other, position))
            index[token].append(position)

    return {(ordered[a].index, ordered[b].index) for a, b in candidates}


def stage2(
    settings: Settings | None = None,
    *,
    start: Quarter | None = None,
    end: Quarter | None = None,
) -> DedupStats:
    """Run probabilistic deduplication over the loaded corpus."""
    settings = settings or get_settings()
    stats = DedupStats()
    root = parquet_root(settings)

    with connect(settings) as handle:
        rows = _load_blocking_input(handle, root, start=start, end=end)
        stats.records_excluded_null_key = _count_excluded(handle, root, start=start, end=end)

    stats.records_considered = len(rows)
    blocks: dict[tuple[str, float, str], list[BlockRecord]] = defaultdict(list)
    for record in rows:
        blocks[record.block_key].append(record)

    stats.blocks = len(blocks)
    seen: dict[int, tuple[int, float, float, bool]] = {}

    for members in blocks.values():
        size = len(members)
        stats.largest_block = max(stats.largest_block, size)
        stats.naive_comparisons += size * (size - 1) // 2
        if size < 2:
            continue

        pairs = (
            _candidate_pairs(members, REACTION_THRESHOLD)
            if size >= BLOCK_INDEX_THRESHOLD
            else {
                (members[a].index, members[b].index)
                for a in range(size)
                for b in range(a + 1, size)
            }
        )

        by_index = {record.index: record for record in members}
        for left_index, right_index in pairs:
            left, right = by_index[left_index], by_index[right_index]
            if left.caseid == right.caseid:
                continue
            stats.comparisons += 1
            reaction_similarity = jaccard(left.reactions, right.reactions)
            if reaction_similarity < REACTION_THRESHOLD:
                continue
            drug_similarity = jaccard(left.drugs, right.drugs)
            # Sex and country are equal by construction: they are the block key.
            if not is_duplicate(
                drug_similarity,
                reaction_similarity,
                same_sex=True,
                age_difference=abs(left.age - right.age),
                same_country=True,
            ):
                continue

            # The most recently received report is kept as canonical.
            survivor, superseded = (left, right) if left.fda_dt >= right.fda_dt else (right, left)
            cross_quarter = left.quarter != right.quarter
            stats.duplicate_pairs += 1
            if cross_quarter:
                stats.cross_quarter_pairs += 1
            seen[superseded.primaryid] = (
                survivor.primaryid,
                drug_similarity,
                reaction_similarity,
                cross_quarter,
            )

    stats.pairs = [
        (primaryid, canonical, drug_similarity, reaction_similarity, cross_quarter)
        for primaryid, (
            canonical,
            drug_similarity,
            reaction_similarity,
            cross_quarter,
        ) in seen.items()
    ]
    log.info(
        "faers.dedup.stage2",
        considered=stats.records_considered,
        excluded=stats.records_excluded_null_key,
        blocks=stats.blocks,
        largest_block=stats.largest_block,
        comparisons=stats.comparisons,
        naive=stats.naive_comparisons,
        duplicates=stats.duplicate_pairs,
        cross_quarter=stats.cross_quarter_pairs,
    )
    return stats


def _quarter_filter(start: Quarter | None, end: Quarter | None) -> str:
    clauses = []
    if start is not None:
        clauses.append(f"c.quarter >= '{start.label}'")
    if end is not None:
        clauses.append(f"c.quarter <= '{end.label}'")
    return (" AND " + " AND ".join(clauses)) if clauses else ""


def _load_blocking_input(
    handle: duckdb.DuckDBPyConnection,
    root: object,
    *,
    start: Quarter | None,
    end: Quarter | None,
) -> list[BlockRecord]:
    """Cases with a complete blocking key, with their drug and reaction sets."""
    where = _quarter_filter(start, end)
    query = f"""
        SELECT c.primaryid, c.caseid, c.quarter, c.sex, c.country,
               round(c.age_years) AS age_r, c.fda_dt,
               coalesce(d.drugs, []) AS drugs, coalesce(r.reacs, []) AS reacs
        FROM read_parquet('{root}/case/*/*.parquet', hive_partitioning := true) c
        LEFT JOIN (
            SELECT primaryid, list_distinct(list(drugname_raw)) AS drugs
            FROM read_parquet('{root}/drug/*/*.parquet', hive_partitioning := true)
            WHERE drugname_raw IS NOT NULL GROUP BY primaryid
        ) d USING (primaryid)
        LEFT JOIN (
            SELECT primaryid, list_distinct(list(pt)) AS reacs
            FROM read_parquet('{root}/reaction/*/*.parquet', hive_partitioning := true)
            WHERE pt IS NOT NULL GROUP BY primaryid
        ) r USING (primaryid)
        WHERE c.sex IS NOT NULL AND c.country IS NOT NULL AND c.age_years IS NOT NULL {where}
    """
    return [
        BlockRecord(
            index=index,
            primaryid=int(primaryid),
            caseid=int(caseid),
            quarter=str(quarter),
            sex=str(sex),
            country=str(country),
            age=float(age),
            fda_dt=str(fda_dt) if fda_dt is not None else "",
            drugs=frozenset(drugs or []),
            reactions=frozenset(reactions or []),
        )
        for index, (
            primaryid,
            caseid,
            quarter,
            sex,
            country,
            age,
            fda_dt,
            drugs,
            reactions,
        ) in enumerate(handle.execute(query).fetchall())
    ]


def _count_excluded(
    handle: duckdb.DuckDBPyConnection,
    root: object,
    *,
    start: Quarter | None,
    end: Quarter | None,
) -> int:
    """Cases that cannot be blocked because a key component is missing."""
    where = _quarter_filter(start, end)
    query = f"""
        SELECT count(*) FROM read_parquet('{root}/case/*/*.parquet', hive_partitioning := true) c
        WHERE (c.sex IS NULL OR c.country IS NULL OR c.age_years IS NULL) {where}
    """
    result = handle.execute(query).fetchone()
    return int(result[0]) if result else 0


def persist(stats: DedupStats, version_pairs: list[tuple[int, int]]) -> int:
    """Replace the duplicate table with the results of one coherent run.

    Truncate and rebuild rather than append: appending would accumulate the
    findings of successive runs with different corpora and different code, and
    there would be no way to tell afterwards which rows came from which.
    """
    from signaldesk.web.signals.models import Duplicate

    Duplicate.objects.all().delete()
    records = [
        Duplicate(
            primaryid=primaryid,
            canonical_primaryid=canonical,
            method=METHOD_VERSION,
            score=1.0,
        )
        for primaryid, canonical in version_pairs
    ]
    superseded = {primaryid for primaryid, _ in version_pairs}
    records.extend(
        Duplicate(
            primaryid=primaryid,
            canonical_primaryid=canonical,
            method=METHOD_PROBABILISTIC,
            score=min(drug_similarity, reaction_similarity),
            drug_jaccard=drug_similarity,
            reaction_jaccard=reaction_similarity,
            cross_quarter=cross_quarter,
        )
        for primaryid, canonical, drug_similarity, reaction_similarity, cross_quarter in stats.pairs
        if primaryid not in superseded
    )
    Duplicate.objects.bulk_create(records, batch_size=10_000)
    log.info("faers.dedup.persisted", rows=len(records))
    return len(records)


def resolve_versions_across_quarters(settings: Settings | None = None) -> list[tuple[int, int]]:
    """Case identifiers appearing at several versions across the whole corpus.

    Returns (superseded primaryid, surviving primaryid) pairs. Within one quarter
    this is handled during ingest; across quarters it can only be done once every
    quarter is loaded, which is why it lives here.
    """
    settings = settings or get_settings()
    root = parquet_root(settings)
    query = f"""
        WITH ranked AS (
            SELECT primaryid, caseid, caseversion,
                   max(caseversion) OVER (PARTITION BY caseid) AS best,
                   first_value(primaryid) OVER (
                       PARTITION BY caseid ORDER BY caseversion DESC, primaryid DESC
                   ) AS survivor
            FROM read_parquet('{root}/case/*/*.parquet', hive_partitioning := true)
        )
        -- DISTINCT because a case republished in a later quarter appears once
        -- per quarter it was published in, and this is a statement about the
        -- case identifier, not about the publications.
        SELECT DISTINCT primaryid, survivor FROM ranked
        WHERE (caseversion IS DISTINCT FROM best OR primaryid <> survivor)
          AND primaryid <> survivor
    """
    with connect(settings) as handle:
        rows = handle.execute(query).fetchall()
    return [(int(primaryid), int(survivor)) for primaryid, survivor in rows]
