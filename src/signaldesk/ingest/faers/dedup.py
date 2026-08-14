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

import json
import math
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path

import duckdb
from django.db import transaction

from signaldesk.core.config import Settings, get_settings
from signaldesk.core.db import connect
from signaldesk.core.logging import get_logger
from signaldesk.ingest.faers.load import parquet_root
from signaldesk.ingest.faers.quarter import Quarter
from signaldesk.web.signals.models import Duplicate

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

#: Rows pulled from the engine at a time while streaming blocks.
FETCH_BATCH_ROWS = 50_000

#: Duplicate rows inserted per batch. A full-corpus run writes millions.
PERSIST_BATCH_ROWS = 10_000


@dataclass(slots=True)
class DedupStats:
    """Everything the quality report needs to describe a dedup run."""

    records_considered: int = 0
    records_excluded_null_key: int = 0
    #: Set when the figures were rebuilt from stored results rather than
    #: measured by a pass, so the report can say which are unavailable.
    from_store: bool = False
    records_flagged: int | None = None
    #: Flagged records split by which rule caught them. The two rules apply to
    #: different populations, so they cannot share a denominator.
    records_flagged_version: int | None = None
    records_flagged_probabilistic: int | None = None
    blocks: int = 0
    largest_block: int = 0
    comparisons: int = 0
    naive_comparisons: int = 0
    duplicate_pairs: int = 0
    cross_quarter_pairs: int = 0
    superseded_versions: int = 0
    pairs: list[tuple[int, int, float, float, bool]] = field(default_factory=list)

    @property
    def duplicate_records(self) -> int:
        """Records judged to duplicate another one.

        Not the same as `duplicate_pairs`, and the difference is large: one
        record can pair with many others, so pairs run to tens of millions while
        the records they implicate run to a few million. A rate computed from
        pairs exceeds 100 percent and means nothing. Rates use this.
        """
        return self.records_flagged if self.records_flagged is not None else len(self.pairs)

    @property
    def cross_quarter_share(self) -> float:
        """Share of detected duplicates whose members sit in different quarters.

        The number that justifies running this corpus-wide: anything above zero
        is invisible to a per-quarter pass.

        Measured by a pass this is a share of pairs. Rebuilt from stored results
        the pair count is unavailable and the numerator is a count of flagged
        records, so the denominator has to match or the share reads as zero.
        """
        denominator = self.duplicate_records if self.from_store else self.duplicate_pairs
        if not denominator:
            return 0.0
        return self.cross_quarter_pairs / denominator

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


def _candidate_pairs(members: list[BlockRecord], threshold: float) -> Iterator[tuple[int, int]]:
    """Yield pairs that could reach the threshold, by size and prefix filtering.

    A generator, not a set. The largest block in the corpus holds 109,667
    records, and collecting its candidates before scoring any of them builds a
    structure with hundreds of millions of entries: memory grows with the square
    of the biggest block, which is the one thing this filtering exists to avoid.
    Yielding keeps the working set to one block's inverted index.

    Duplicate suggestions from two shared prefix tokens are suppressed per
    record with a small set bounded by the block, rather than globally.
    """
    ordered = sorted(members, key=lambda record: len(record.reactions))
    index: dict[str, list[int]] = defaultdict(list)

    for position, record in enumerate(ordered):
        size = len(record.reactions)
        if size == 0:
            continue
        lower_bound = size * threshold
        prefix = sorted(record.reactions)[: prefix_length(size, threshold)]
        suggested: set[int] = set()
        for token in prefix:
            for other in index[token]:
                if other in suggested:
                    continue
                suggested.add(other)
                if len(ordered[other].reactions) >= lower_bound:
                    yield (ordered[other].index, ordered[position].index)
            index[token].append(position)


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
    seen: dict[int, tuple[int, float, float, bool]] = {}

    with connect(settings) as handle:
        stats.records_excluded_null_key = _count_excluded(handle, root, start=start, end=end)
        _stage_input(handle, root, start=start, end=end)

        for members in _blocks(handle):
            stats.records_considered += len(members)
            stats.blocks += 1
            _compare_block(members, stats, seen)

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


def _compare_block(
    members: list[BlockRecord],
    stats: DedupStats,
    seen: dict[int, tuple[int, float, float, bool]],
) -> None:
    """Score one block, recording the duplicates it contains."""
    size = len(members)
    stats.largest_block = max(stats.largest_block, size)
    stats.naive_comparisons += size * (size - 1) // 2
    if size < 2:
        return

    pairs = (
        _candidate_pairs(members, REACTION_THRESHOLD)
        if size >= BLOCK_INDEX_THRESHOLD
        else {(members[a].index, members[b].index) for a in range(size) for b in range(a + 1, size)}
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


def _quarter_filter(start: Quarter | None, end: Quarter | None) -> str:
    clauses = []
    if start is not None:
        clauses.append(f"c.quarter >= '{start.label}'")
    if end is not None:
        clauses.append(f"c.quarter <= '{end.label}'")
    return (" AND " + " AND ".join(clauses)) if clauses else ""


def _stage_input(
    handle: duckdb.DuckDBPyConnection,
    root: object,
    *,
    start: Quarter | None,
    end: Quarter | None,
) -> None:
    """Materialise the deduplication input to disk, in stages.

    Twenty million cases joined against fifty million drug rows and forty-five
    million reaction rows, collapsed into per-case sets, does not fit in memory,
    and no memory limit makes it fit: the working set grows with the corpus.

    So it is built as tables rather than returned as a result. Each CREATE TABLE
    streams its output to the database file instead of accumulating it for the
    client, and the expensive grouping happens once rather than once per read.
    """
    where = _quarter_filter(start, end)

    handle.execute("DROP TABLE IF EXISTS dedup_input")
    handle.execute("DROP TABLE IF EXISTS dedup_drug")
    handle.execute("DROP TABLE IF EXISTS dedup_reaction")

    log.info("faers.dedup.staging", step="drug_sets")
    handle.execute(f"""
        CREATE TABLE dedup_drug AS
        SELECT primaryid, list_distinct(list(drugname_raw)) AS drugs
        FROM read_parquet('{root}/drug/*/*.parquet', hive_partitioning := true)
        WHERE drugname_raw IS NOT NULL
        GROUP BY primaryid
    """)

    log.info("faers.dedup.staging", step="reaction_sets")
    handle.execute(f"""
        CREATE TABLE dedup_reaction AS
        SELECT primaryid, list_distinct(list(pt)) AS reactions
        FROM read_parquet('{root}/reaction/*/*.parquet', hive_partitioning := true)
        WHERE pt IS NOT NULL
        GROUP BY primaryid
    """)

    log.info("faers.dedup.staging", step="blockable_cases")
    handle.execute(f"""
        CREATE TABLE dedup_input AS
        SELECT c.primaryid, c.caseid, c.quarter, c.sex, c.country,
               round(c.age_years) AS age_r, coalesce(c.fda_dt::VARCHAR, '') AS fda_dt,
               coalesce(d.drugs, []) AS drugs, coalesce(r.reactions, []) AS reactions
        FROM read_parquet('{root}/case/*/*.parquet', hive_partitioning := true) c
        LEFT JOIN dedup_drug d USING (primaryid)
        LEFT JOIN dedup_reaction r USING (primaryid)
        WHERE c.sex IS NOT NULL AND c.country IS NOT NULL AND c.age_years IS NOT NULL {where}
        QUALIFY count(*) OVER (PARTITION BY c.sex, round(c.age_years), c.country) > 1
    """)
    handle.execute("DROP TABLE dedup_drug")
    handle.execute("DROP TABLE dedup_reaction")


def _blocks(handle: duckdb.DuckDBPyConnection) -> Iterator[list[BlockRecord]]:
    """Yield one block at a time from the staged input.

    Read in coarse groups of (sex, country) and sorted by age within the group,
    so each query returns a bounded slice and the blocks inside it arrive
    together. Sorting the whole staged table at once is the operation that runs
    the engine out of memory; sorting one country's records does not.
    """
    groups = handle.execute(
        "SELECT DISTINCT sex, country FROM dedup_input ORDER BY sex, country"
    ).fetchall()

    index = 0
    for sex, country in groups:
        handle.execute(
            "SELECT primaryid, caseid, quarter, sex, country, age_r, fda_dt, drugs, reactions "
            "FROM dedup_input WHERE sex = ? AND country = ? ORDER BY age_r",
            [sex, country],
        )
        current: list[BlockRecord] = []
        current_key: tuple[str, float, str] | None = None

        while batch := handle.fetchmany(FETCH_BATCH_ROWS):
            for row in batch:
                record = BlockRecord(
                    index=index,
                    primaryid=int(row[0]),
                    caseid=int(row[1]),
                    quarter=str(row[2]),
                    sex=str(row[3]),
                    country=str(row[4]),
                    age=float(row[5]),
                    fda_dt=str(row[6]),
                    drugs=frozenset(row[7] or []),
                    reactions=frozenset(row[8] or []),
                )
                index += 1
                if current_key is not None and record.block_key != current_key:
                    yield current
                    current = []
                current_key = record.block_key
                current.append(record)

        if current:
            yield current


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


def _duplicate_rows(stats: DedupStats, version_pairs: list[tuple[int, int]]) -> Iterator[Duplicate]:
    """Yield the rows one run produced, without materialising them all."""
    for primaryid, canonical in version_pairs:
        yield Duplicate(
            primaryid=primaryid,
            canonical_primaryid=canonical,
            method=METHOD_VERSION,
            score=1.0,
        )

    superseded = {primaryid for primaryid, _ in version_pairs}
    for primaryid, canonical, drug_similarity, reaction_similarity, cross_quarter in stats.pairs:
        if primaryid in superseded:
            continue
        yield Duplicate(
            primaryid=primaryid,
            canonical_primaryid=canonical,
            method=METHOD_PROBABILISTIC,
            score=min(drug_similarity, reaction_similarity),
            drug_jaccard=drug_similarity,
            reaction_jaccard=reaction_similarity,
            cross_quarter=cross_quarter,
        )


def stats_from_store(settings: Settings | None = None) -> DedupStats:
    """Rebuild what can be rebuilt from the stored corpus and duplicate table.

    Everything except the pass-only figures - block count, largest block, and
    the two comparison counts - is recoverable: the corpus is on disk and the
    judgements are in the database. Those four are marked unavailable rather
    than defaulted, because a zero there would read as a measurement.
    """
    settings = settings or get_settings()
    root = parquet_root(settings)

    with connect(settings) as handle:
        considered = handle.execute(
            f"""
            SELECT count(*) FROM read_parquet('{root}/case/*/*.parquet', hive_partitioning := true)
            WHERE sex IS NOT NULL AND country IS NOT NULL AND age_years IS NOT NULL
            """
        ).fetchone()
        excluded = _count_excluded(handle, root, start=None, end=None)

    flagged = Duplicate.objects.values("primaryid").distinct().count()
    by_version = Duplicate.objects.filter(method=METHOD_VERSION).count()
    by_probability = Duplicate.objects.filter(method=METHOD_PROBABILISTIC).count()
    cross_quarter = Duplicate.objects.filter(cross_quarter=True).count()
    log.info("faers.dedup.stats_from_store", flagged=flagged, cross_quarter=cross_quarter)

    return DedupStats(
        records_considered=int(considered[0]) if considered else 0,
        records_excluded_null_key=excluded,
        cross_quarter_pairs=cross_quarter,
        records_flagged=flagged,
        records_flagged_version=by_version,
        records_flagged_probabilistic=by_probability,
        from_store=True,
    )


def stats_path(settings: Settings | None = None) -> Path:
    """Where the last deduplication run's measurements are kept."""
    settings = settings or get_settings()
    return settings.data_dir / "faers" / "dedup_stats.json"


def save_stats(stats: DedupStats, settings: Settings | None = None) -> Path:
    """Record what the pass measured, so a report can be rebuilt without it.

    A full-corpus pass costs an hour. Without this, regenerating the quality
    report means paying that again, which is how a report ends up being edited
    by hand instead of produced by a run.
    """
    path = stats_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "measured_at": datetime.now(tz=UTC).isoformat(),
        "records_considered": stats.records_considered,
        "records_excluded_null_key": stats.records_excluded_null_key,
        "blocks": stats.blocks,
        "largest_block": stats.largest_block,
        "comparisons": stats.comparisons,
        "naive_comparisons": stats.naive_comparisons,
        "duplicate_pairs": stats.duplicate_pairs,
        "cross_quarter_pairs": stats.cross_quarter_pairs,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log.info("faers.dedup.stats_saved", path=str(path))
    return path


def load_stats(settings: Settings | None = None) -> DedupStats | None:
    """The last run's measurements, or None if no pass has been recorded."""
    path = stats_path(settings)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        log.warning("faers.dedup.stats_unreadable", path=str(path))
        return None
    return DedupStats(
        records_considered=int(payload["records_considered"]),
        records_excluded_null_key=int(payload["records_excluded_null_key"]),
        blocks=int(payload["blocks"]),
        largest_block=int(payload["largest_block"]),
        comparisons=int(payload["comparisons"]),
        naive_comparisons=int(payload["naive_comparisons"]),
        duplicate_pairs=int(payload["duplicate_pairs"]),
        cross_quarter_pairs=int(payload["cross_quarter_pairs"]),
    )


def persist(
    stats: DedupStats,
    version_pairs: list[tuple[int, int]],
    settings: Settings | None = None,
) -> int:
    """Replace the duplicate table with the results of one coherent run.

    Rebuild rather than append: appending would accumulate the findings of
    successive runs over different corpora and different code, with no way to
    tell afterwards which rows came from which.

    `settings` is threaded through rather than defaulted, because the stats file
    it writes is real state: a caller pointed at a scratch directory, such as a
    test, must not overwrite the record of the last production pass.

    Two properties this has to have, both learned by not having them. The whole
    replacement is one transaction, so a failure leaves the previous results in
    place instead of an empty table. And rows are streamed in batches rather
    than built into one list: a full-corpus run produces about four and a half
    million of them, and holding that many model instances at once exhausts the
    process after an hour of work has already been done.
    """
    written = 0
    with transaction.atomic():
        Duplicate.objects.all().delete()
        rows = _duplicate_rows(stats, version_pairs)
        while batch := list(islice(rows, PERSIST_BATCH_ROWS)):
            Duplicate.objects.bulk_create(batch, batch_size=PERSIST_BATCH_ROWS)
            written += len(batch)
    save_stats(stats, settings)
    log.info("faers.dedup.persisted", rows=written)
    return written


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
