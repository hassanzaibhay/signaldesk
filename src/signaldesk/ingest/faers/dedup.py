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
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path

import duckdb
from django.db import transaction

from signaldesk.core.config import Settings, get_settings
from signaldesk.core.db import connect
from signaldesk.core.errors import IngestError
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
    #: Records stage 1 already resolved. They are not compared: a match on a
    #: record the published rule has withdrawn cannot reach the numerator, so
    #: leaving them in the population would put them in the denominator alone.
    records_excluded_superseded: int = 0
    #: Records with no usable drug string. ``jaccard`` treats an empty set as
    #: dissimilar, so these can never match either, for the same reason.
    records_excluded_empty_drug_set: int = 0
    #: In the population and eligible, but alone in their block, so nothing was
    #: ever compared against them. Named rather than left implicit: the gap
    #: between this count and the staged count is what made the pass and the
    #: store disagree in P02.
    records_excluded_single_member_block: int = 0
    #: Probabilistic matches dropped at write time because the flagged record was
    #: already superseded. Zero once the population excludes them, and recorded
    #: rather than inferred so the quantity survives in stored state.
    probabilistic_discarded_superseded: int | None = None
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
        """Records flagged by either rule.

        Not the same as `duplicate_pairs`, and the difference is large: one
        record can pair with many others, so pairs run to tens of millions while
        the records they implicate run to a few million. A rate computed from
        pairs exceeds 100 percent and means nothing. Rates use this.

        Set by whatever wrote the duplicate table, so it means the same thing on
        both paths. It used to fall back to `len(self.pairs)`, which counts only
        the probabilistic side, so a report built from a measured pass divided by
        a total that excluded every stage 1 flag: the first P03 artifact printed a
        stage 2 numerator of zero and a surviving count 2.85 million too high.
        There is no fallback now - an unset count raises rather than reporting a
        number that means something else.
        """
        if self.records_flagged is None:
            message = (
                "flag counts were never recorded on these statistics, so no rate "
                "can be computed from them"
            )
            raise IngestError(message)
        return self.records_flagged

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
    def population(self) -> int:
        """The denominator a stage 2 rate is reported against.

        Cases the rule can actually apply to: they survive stage 1, carry a
        complete blocking key, and have a drug set to compare. A case alone in
        its block is in the population and simply finds no partner, which is a
        legitimate zero rather than an exclusion, so it is added back explicitly
        instead of being left as a gap between the pass and the store.

        The three exclusions are outside this number by design. A superseded
        record, an unblockable record and a record with no drugs cannot reach the
        numerator, so counting them in the denominator reports a rate against a
        population the rule was never applied to. That was P02's defect.
        """
        return self.records_considered + self.records_excluded_single_member_block

    @property
    def records_evaluated(self) -> int:
        """Every case the pass looked at, whatever became of it."""
        return (
            self.population
            + self.records_excluded_null_key
            + self.records_excluded_superseded
            + self.records_excluded_empty_drug_set
        )

    @property
    def excluded_share(self) -> float:
        """Share of everything looked at that could not be blocked."""
        total = self.records_evaluated
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


def survivor_key(record: BlockRecord) -> tuple[str, int]:
    """The order that decides which of two matching records is kept.

    Most recently received first, then the higher identifier. The second term is
    not decoration: ``fda_dt`` ties are common, and an order that is only a
    partial order lets argument position decide the survivor. Position is
    scoring order, which is not data, and a direction chosen that way can
    disagree with the direction stage 1 chose for the same records. That is one
    of the two arbitrary choices behind the 17 closed components in P02, which
    removed 38 cases with no survivor between them; the other is the canonical
    kept in ``_compare_block``. Which component came from which is not
    recoverable - see ADR 0005.

    ``primaryid`` is unique per record, so this is a total order. Note what that
    does not buy: stage 1 orders on ``(caseversion, primaryid)`` and both edge
    kinds land in one graph, so acyclicity comes from the population filter
    excluding stage-1-superseded records, not from this ordering. ADR 0005 has
    the argument.
    """
    return (record.fda_dt, record.primaryid)


def choose_survivor(left: BlockRecord, right: BlockRecord) -> tuple[BlockRecord, BlockRecord]:
    """Return ``(survivor, superseded)``, independent of argument position."""
    if survivor_key(left) > survivor_key(right):
        return left, right
    return right, left


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
    best: dict[int, tuple[str, int]] = {}

    with connect(settings) as handle:
        counts = stage_population(handle, root, start=start, end=end)
        _apply_counts(stats, counts)

        streamed = 0
        for members in _blocks(handle):
            stats.blocks += 1
            streamed += len(members)
            _compare_block(members, stats, seen, best)

    # The denominator is counted in SQL and the comparison reads the same table.
    # If those two disagree the rate is computed over a population that was not
    # the one compared, which is the failure this whole pass exists to avoid.
    if streamed != counts.compared:
        message = f"staged {counts.compared} records but streamed {streamed}"
        raise IngestError(message)

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
    best: dict[int, tuple[str, int]],
) -> None:
    """Score one block, recording the duplicates it contains.

    `best` holds the survivor key already recorded for each flagged record, so a
    record matching several others keeps the strongest of them rather than
    whichever comparison ran last. Without it the canonical was decided by
    scoring order: the same arbitrary-choice defect as the tie-break, one level
    up.

    The selection is a single hop over the partners this record actually
    matched, not over the block. Jaccard is not transitive, so a canonical may
    itself be flagged against a record the first one never matched, and chains
    through a block are a normal outcome rather than a defect: a block is not an
    equivalence class, and resolving transitively to one representative would
    merge records on the strength of a match neither made. What this fixes is
    the arbitrariness, not the shape - the pointer is now a function of the data,
    and two passes over one corpus write the same one.
    """
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
        survivor, superseded = choose_survivor(left, right)
        cross_quarter = left.quarter != right.quarter
        stats.duplicate_pairs += 1
        if cross_quarter:
            stats.cross_quarter_pairs += 1
        key = survivor_key(survivor)
        recorded = best.get(superseded.primaryid)
        if recorded is not None and key <= recorded:
            continue
        best[superseded.primaryid] = key
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


def block_key_sql(alias: str = "") -> str:
    """The block key in SQL, written once so no caller can spell it differently.

    ``(sex, rounded age, country)``. The staging QUALIFY and every count over the
    population share this, because a partition expression that differs by a
    ``round`` in one place is a defect nothing would surface.
    """
    prefix = f"{alias}." if alias else ""
    return f"{prefix}sex, round({prefix}age_years), {prefix}country"


@dataclass(frozen=True, slots=True)
class PopulationCounts:
    """How every case the pass looked at was classified.

    The four terms partition ``evaluated`` exactly. A rate reported against
    anything other than ``eligible`` is reported against a population the rule
    was not applied to.
    """

    evaluated: int
    eligible: int
    compared: int
    single_member_block: int
    excluded_superseded: int
    excluded_null_key: int
    excluded_empty_drug_set: int


def _survivors_sql(source: str) -> str:
    """Cross-quarter stage 1 survivors: the highest version of each case.

    One definition, used by the population builder and by
    ``resolve_versions_across_quarters``. Two copies of this rule that drift
    apart would put a record in one stage's output and the other's input.
    """
    return f"""
        SELECT primaryid, caseid, caseversion,
               first_value(primaryid) OVER (
                   PARTITION BY caseid ORDER BY caseversion DESC, primaryid DESC
               ) AS survivor
        FROM {source}
    """


def stage_population(
    handle: duckdb.DuckDBPyConnection,
    root: object,
    *,
    start: Quarter | None = None,
    end: Quarter | None = None,
) -> PopulationCounts:
    """Build the deduplication input on disk and count what did not reach it.

    Twenty million cases joined against eighty-four million drug rows and
    sixty-six million reaction rows, collapsed into per-case sets, does not fit
    in memory, and no memory limit makes it fit: the working set grows with the
    corpus. So it is built as tables rather than returned as a result. Each
    CREATE TABLE streams to the database file instead of accumulating a result
    for the client.

    Counting happens here rather than in the caller because the pass and the
    report must not compute the population two different ways. In P02 they did,
    and the store-derived denominator exceeded the compared one by 4,646 with
    nothing in the code saying so.

    Three classes of case are excluded, each because it cannot reach the
    numerator and would otherwise sit in the denominator alone:

    * **superseded** - stage 1 already resolved it to a later version;
    * **null key** - it cannot be blocked, and a null key is not evidence of
      similarity;
    * **empty drug set** - ``jaccard`` scores an empty set as dissimilar, so it
      can never match.

    A case alone in its block is *not* excluded. The rule applies to it and finds
    nothing, which is a legitimate zero, so it stays in the denominator and is
    reported separately.
    """
    where = _quarter_filter(start, end)

    staged = ("dedup_input", "dedup_drug", "dedup_reaction", "dedup_survivor", "dedup_status")
    for table in (*staged, "dedup_case"):
        handle.execute(f"DROP TABLE IF EXISTS {table}")

    # One row per case, not one per publication. The analytical store keeps a
    # row for each quarter a case was published in, 707 of them corpus-wide, and
    # every count downstream is a count of cases. Collapsing here means the
    # population and the flags are counting the same kind of thing, and the
    # totals agree with Postgres rather than exceeding it. See ADR 0004.
    log.info("faers.dedup.staging", step="cases")
    handle.execute(f"""
        CREATE TABLE dedup_case AS
        SELECT c.primaryid, c.caseid, c.caseversion, c.quarter, c.sex, c.country,
               c.age_years, coalesce(c.fda_dt::VARCHAR, '') AS fda_dt
        FROM read_parquet('{root}/case/*/*.parquet', hive_partitioning := true) c
        WHERE 1 = 1 {where}
        QUALIFY row_number() OVER (PARTITION BY c.primaryid ORDER BY c.quarter DESC) = 1
    """)

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

    log.info("faers.dedup.staging", step="stage1_survivors")
    handle.execute(f"""
        CREATE TABLE dedup_survivor AS
        SELECT DISTINCT primaryid FROM ({_survivors_sql("dedup_case")}) ranked
        WHERE primaryid = survivor
    """)

    log.info("faers.dedup.staging", step="classify")
    handle.execute("""
        CREATE TABLE dedup_status AS
        SELECT c.primaryid,
               CASE
                   WHEN s.primaryid IS NULL THEN 'superseded'
                   WHEN c.sex IS NULL OR c.country IS NULL OR c.age_years IS NULL
                       THEN 'null_key'
                   WHEN d.primaryid IS NULL THEN 'empty_drug_set'
                   ELSE 'eligible'
               END AS status
        FROM dedup_case c
        LEFT JOIN dedup_survivor s USING (primaryid)
        LEFT JOIN dedup_drug d USING (primaryid)
    """)

    log.info("faers.dedup.staging", step="blockable_cases")
    handle.execute(f"""
        CREATE TABLE dedup_input AS
        SELECT c.primaryid, c.caseid, c.quarter, c.sex, c.country,
               round(c.age_years) AS age_r, c.fda_dt,
               coalesce(d.drugs, []) AS drugs, coalesce(r.reactions, []) AS reactions
        FROM dedup_case c
        JOIN dedup_status st ON st.primaryid = c.primaryid AND st.status = 'eligible'
        LEFT JOIN dedup_drug d ON d.primaryid = c.primaryid
        LEFT JOIN dedup_reaction r ON r.primaryid = c.primaryid
        QUALIFY count(*) OVER (PARTITION BY {block_key_sql("c")}) > 1
    """)
    handle.execute("DROP TABLE dedup_drug")
    handle.execute("DROP TABLE dedup_reaction")

    return _population_counts(handle)


def _population_counts(handle: duckdb.DuckDBPyConnection) -> PopulationCounts:
    """Read the classification back, from the tables the pass will compare."""
    by_status = dict(
        handle.execute("SELECT status, count(*) FROM dedup_status GROUP BY status").fetchall()
    )
    compared_row = handle.execute("SELECT count(*) FROM dedup_input").fetchone()
    compared = int(compared_row[0]) if compared_row else 0
    eligible = int(by_status.get("eligible", 0))
    counts = PopulationCounts(
        evaluated=sum(int(value) for value in by_status.values()),
        eligible=eligible,
        compared=compared,
        single_member_block=eligible - compared,
        excluded_superseded=int(by_status.get("superseded", 0)),
        excluded_null_key=int(by_status.get("null_key", 0)),
        excluded_empty_drug_set=int(by_status.get("empty_drug_set", 0)),
    )
    log.info(
        "faers.dedup.population",
        evaluated=counts.evaluated,
        eligible=counts.eligible,
        compared=counts.compared,
        alone_in_block=counts.single_member_block,
        superseded=counts.excluded_superseded,
        null_key=counts.excluded_null_key,
        empty_drug_set=counts.excluded_empty_drug_set,
    )
    return counts


def _apply_counts(stats: DedupStats, counts: PopulationCounts) -> None:
    """Copy a population classification onto the run's statistics."""
    stats.records_considered = counts.compared
    stats.records_excluded_single_member_block = counts.single_member_block
    stats.records_excluded_superseded = counts.excluded_superseded
    stats.records_excluded_null_key = counts.excluded_null_key
    stats.records_excluded_empty_drug_set = counts.excluded_empty_drug_set


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


def _duplicate_rows(stats: DedupStats, version_pairs: list[tuple[int, int]]) -> Iterator[Duplicate]:
    """Yield the rows one run produced, without materialising them all.

    The skip below is a backstop, not the mechanism. Stage 2's population
    excludes stage 1 superseded records, so nothing should reach it; the counter
    exists to say that rather than leave it assumed, and a non-zero value means
    the population filter is not doing what it claims.
    """
    for primaryid, canonical in version_pairs:
        yield Duplicate(
            primaryid=primaryid,
            canonical_primaryid=canonical,
            method=METHOD_VERSION,
            score=1.0,
        )

    superseded = {primaryid for primaryid, _ in version_pairs}
    discarded = 0
    stats.probabilistic_discarded_superseded = discarded
    for primaryid, canonical, drug_similarity, reaction_similarity, cross_quarter in stats.pairs:
        if primaryid in superseded:
            discarded += 1
            stats.probabilistic_discarded_superseded = discarded
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

    The population is rebuilt by the same ``stage_population`` the pass runs,
    not by a second query that means to say the same thing. In P02 it was the
    second query, it omitted the single-member-block filter, and the store's
    denominator exceeded the compared one by 4,646 with nothing to catch it.
    """
    settings = settings or get_settings()
    root = parquet_root(settings)

    with connect(settings) as handle:
        counts = stage_population(handle, root)

    flagged = Duplicate.objects.values("primaryid").distinct().count()
    by_version = Duplicate.objects.filter(method=METHOD_VERSION).count()
    by_probability = Duplicate.objects.filter(method=METHOD_PROBABILISTIC).count()
    cross_quarter = Duplicate.objects.filter(cross_quarter=True).count()
    log.info("faers.dedup.stats_from_store", flagged=flagged, cross_quarter=cross_quarter)

    stats = DedupStats(
        cross_quarter_pairs=cross_quarter,
        records_flagged=flagged,
        records_flagged_version=by_version,
        records_flagged_probabilistic=by_probability,
        from_store=True,
    )
    _apply_counts(stats, counts)
    return stats


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
        "records_excluded_superseded": stats.records_excluded_superseded,
        "records_excluded_empty_drug_set": stats.records_excluded_empty_drug_set,
        "records_excluded_single_member_block": stats.records_excluded_single_member_block,
        "probabilistic_discarded_superseded": stats.probabilistic_discarded_superseded,
        # Carried so a report rebuilt from this file divides by the same counts
        # the pass wrote, rather than falling back to something that counts one
        # rule and calls it the total.
        "records_flagged": stats.records_flagged,
        "records_flagged_version": stats.records_flagged_version,
        "records_flagged_probabilistic": stats.records_flagged_probabilistic,
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
    discarded = payload.get("probabilistic_discarded_superseded")
    flagged = payload.get("records_flagged")
    flagged_version = payload.get("records_flagged_version")
    flagged_probabilistic = payload.get("records_flagged_probabilistic")
    return DedupStats(
        records_flagged=None if flagged is None else int(flagged),
        records_flagged_version=None if flagged_version is None else int(flagged_version),
        records_flagged_probabilistic=(
            None if flagged_probabilistic is None else int(flagged_probabilistic)
        ),
        records_considered=int(payload["records_considered"]),
        records_excluded_null_key=int(payload["records_excluded_null_key"]),
        # Defaulted rather than required: a stats file written by the previous
        # pass predates these counts, and a missing count is not a zero one.
        records_excluded_superseded=int(payload.get("records_excluded_superseded", 0)),
        records_excluded_empty_drug_set=int(payload.get("records_excluded_empty_drug_set", 0)),
        records_excluded_single_member_block=int(
            payload.get("records_excluded_single_member_block", 0)
        ),
        probabilistic_discarded_superseded=None if discarded is None else int(discarded),
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

    # The flag counts are recorded by whatever wrote the table, which is here.
    # Every rate downstream divides by them, and a pass that leaves them unset
    # produces a report whose numerator counts one rule and whose total counts
    # neither.
    stats.records_flagged_version = len(version_pairs)
    stats.records_flagged_probabilistic = written - len(version_pairs)
    stats.records_flagged = written
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
    source = f"read_parquet('{root}/case/*/*.parquet', hive_partitioning := true)"
    query = f"""
        WITH ranked AS ({_survivors_sql(source)})
        -- DISTINCT because a case republished in a later quarter appears once
        -- per quarter it was published in, and this is a statement about the
        -- case identifier, not about the publications.
        SELECT DISTINCT primaryid, survivor FROM ranked
        WHERE primaryid <> survivor
    """
    with connect(settings) as handle:
        rows = handle.execute(query).fetchall()
    return [(int(primaryid), int(survivor)) for primaryid, survivor in rows]


@dataclass(frozen=True, slots=True)
class ChainReport:
    """What resolving every canonical pointer to its root found.

    The surviving corpus is derived as total minus flagged, and that subtraction
    assumes every flagged record has a surviving representative: that following
    canonical pointers from it ends at a record nothing flagged. A component in
    which every member is flagged breaks the assumption, and the cases in it
    leave the corpus rather than being merged into a survivor.

    Both counts must be zero before any duplicate rate is reported.
    """

    edges: int
    nodes: int
    flagged: int
    unflagged: int
    resolved_to_unflagged: int
    resolved_into_cycle: int
    components: int
    closed_components: int
    records_in_closed_components: int
    cycles: int
    records_in_cycles: int
    cycle_lengths: tuple[int, ...]
    closed_component_members: tuple[tuple[int, ...], ...]

    @property
    def is_sound(self) -> bool:
        """True when every chain ends at an unflagged record."""
        return self.closed_components == 0 and self.cycles == 0


def resolve_chains(edges: Iterable[tuple[int, int]]) -> ChainReport:
    """Follow every canonical pointer to its root.

    Pure over the edge list so it can be tested on a hand-built graph with no
    database. Each flagged record carries exactly one canonical, which makes this
    a functional graph: a component with no unflagged member necessarily contains
    a cycle. Both are counted independently anyway, because agreeing is evidence
    and assuming is not.

    A record with two outgoing pointers means one record was flagged under two
    rules, which the population rules forbid, so it raises rather than being
    resolved arbitrarily.
    """
    successor: dict[int, int] = {}
    for primaryid, canonical in edges:
        if primaryid in successor and successor[primaryid] != canonical:
            message = (
                f"record {primaryid} is flagged twice, against {successor[primaryid]} "
                f"and {canonical}"
            )
            raise IngestError(message)
        successor[primaryid] = canonical

    nodes = set(successor) | set(successor.values())
    cycles = _find_cycles(successor)
    in_cycle = {member for cycle in cycles for member in cycle}
    roots = _resolve_roots(successor, in_cycle)

    parent = _union_components(successor, nodes)
    members: dict[int, list[int]] = defaultdict(list)
    unflagged_in_component: set[int] = set()
    for node in nodes:
        root = _find(parent, node)
        members[root].append(node)
        if node not in successor:
            unflagged_in_component.add(root)

    closed = tuple(
        tuple(sorted(group))
        for root, group in sorted(members.items())
        if root not in unflagged_in_component
    )
    resolved_to_unflagged = sum(1 for node in successor if roots[node] not in successor)
    return ChainReport(
        edges=len(successor),
        nodes=len(nodes),
        flagged=len(successor),
        unflagged=len(nodes) - len(successor),
        resolved_to_unflagged=resolved_to_unflagged,
        resolved_into_cycle=len(successor) - resolved_to_unflagged,
        components=len(members),
        closed_components=len(closed),
        records_in_closed_components=sum(len(group) for group in closed),
        cycles=len(cycles),
        records_in_cycles=len(in_cycle),
        cycle_lengths=tuple(sorted(len(cycle) for cycle in cycles)),
        closed_component_members=closed,
    )


def _find_cycles(successor: dict[int, int]) -> list[list[int]]:
    """Every cycle in the pointer graph, walked iteratively.

    Recursion is not an option: a chain can be as long as the corpus, and the
    interpreter's stack is not.
    """
    unvisited, on_path, done = 0, 1, 2
    state: dict[int, int] = {}
    cycles: list[list[int]] = []

    for start in successor:
        if state.get(start, unvisited) != unvisited:
            continue
        path: list[int] = []
        node = start
        while True:
            current = state.get(node, unvisited)
            if current == done or node not in successor:
                break
            if current == on_path:
                cycles.append(path[path.index(node) :])
                break
            state[node] = on_path
            path.append(node)
            node = successor[node]
        for member in path:
            state[member] = done
    return cycles


def _resolve_roots(successor: dict[int, int], in_cycle: set[int]) -> dict[int, int]:
    """Where each flagged record's chain ends, memoised along the way.

    A record on a cycle is its own root: its chain ends nowhere, which is the
    condition this exists to detect, so it is seeded rather than walked.
    """
    roots: dict[int, int] = {member: member for member in in_cycle}
    for start in successor:
        if start in roots:
            continue
        path: list[int] = []
        node = start
        while node not in roots and node in successor:
            path.append(node)
            node = successor[node]
        end = roots.get(node, node)
        for member in path:
            roots[member] = end
    return roots


def _union_components(successor: dict[int, int], nodes: set[int]) -> dict[int, int]:
    """Union-find over the edges, ignoring their direction."""
    parent = {node: node for node in nodes}
    for left, right in successor.items():
        left_root, right_root = _find(parent, left), _find(parent, right)
        if left_root != right_root:
            parent[left_root] = right_root
    return parent


def _find(parent: dict[int, int], node: int) -> int:
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = parent[node]
    return node


def chain_report(settings: Settings | None = None) -> ChainReport:
    """Resolve the stored duplicate table's pointers. The gate before a delta."""
    settings = settings or get_settings()
    edges = Duplicate.objects.values_list("primaryid", "canonical_primaryid").iterator(
        chunk_size=PERSIST_BATCH_ROWS
    )
    report = resolve_chains(edges)
    log.info(
        "faers.dedup.chains",
        flagged=report.flagged,
        components=report.components,
        closed_components=report.closed_components,
        cycles=report.cycles,
        records_in_closed_components=report.records_in_closed_components,
        sound=report.is_sound,
    )
    return report
