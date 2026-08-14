# 4. Split serving from analytics for the case child tables

Date: 2026-08-13

## Status

Accepted

## Context

The adverse event corpus was to be held twice: in Parquet for analysis and in
Postgres for the application to query. Measuring the first five quarters showed
what that costs.

Loaded: 1,050,931 cases across 2012Q4 to 2013Q4.

| Store | Measured | Projected, 55 quarters |
|---|---|---|
| Parquet, all datasets | 56.7 MB | 0.96 GB |
| Postgres, all tables | 1,845.8 MB | 31.18 GB |

The projection scales by cases rather than by quarter count, because quarters
grow: the 2013 quarters carry about 223,000 cases each and 2026Q2 carries
422,459, giving roughly 17.8 million cases over the full range.

Adding 10 GB of operating headroom puts the requirement at **42.13 GB against
36.98 GB of free disk**, after reclaiming the build cache, on a machine with one
volume. The full corpus does not fit.

The Postgres figure is dominated by the child tables, and by their indexes:

| Table | Size with indexes |
|---|---|
| faers_reaction | 545 MB |
| faers_drug | 482 MB |
| faers_indication | 233 MB |
| faers_case | 227 MB |
| faers_therapy | 182 MB |
| faers_outcome | 81 MB |

## Decision

Postgres retains `faers_case`, `faers_duplicate`, `ingest_manifest`, and the
curated and derived tables that later work adds.

The five high-cardinality child datasets - drug, reaction, outcome, therapy and
indication - are held in Parquet only and read through DuckDB. Typed accessors
live in `signaldesk.analytics.faers` so that no caller reaches for a Django
model that does not exist, and so the partition layout stays a detail of that
module.

Case-level seriousness survives the split: `serious`, `serious_death` and the
`outcome_codes` evidence list are stored on the case itself, so the outcome
dataset is not needed to answer a question about a case.

**This is a serving-versus-analytics split, not a scope cut.** Every row is still
ingested, still validated, still queryable, and still counted in the quality
report. What changes is which engine answers the question. Nothing in the
application serves those tables row by row: every consumer of them is an
aggregate, and the contingency builder reads Parquet regardless because a
full-corpus cross-tabulation is what the analytical engine is for.

## Measured outcome

The projection above was made from five quarters. The full corpus has since been
ingested, so the estimate can be replaced with the real figures.

| | Projected | Measured |
|---|---|---|
| Cases | 17,750,562 | 20,534,506 |
| Postgres, after the split | 4.48 GB | 4.95 GB |
| Parquet, all datasets | 0.96 GB | 1.20 GB |
| Total, excluding headroom | 5.44 GB | 6.15 GB |

The corpus is 16 percent larger than projected, because quarter sizes were
extrapolated linearly between the earliest and latest quarter and the growth is
steeper than that. The split still achieves what it was adopted for: 6.15 GB
against 36.98 GB free, where the unsplit design projected 42.13 GB.

Row counts in the analytical store, from a single glob across all 55 partitions:
84.6 million drug rows, 66.3 million reactions, 55.8 million indications, 29.3
million therapies, 15.2 million outcomes, 0.9 million report sources.

One difference worth knowing when reconciling the two stores: Parquet holds
20,535,213 case rows against Postgres's 20,534,506. Parquet records one row per
publication and Postgres one row per case, and the 707-row gap is cases the
source republished in a later quarter.

## Consequences

Projected Postgres for the full corpus falls from 31.18 GB to about 4.2 GB, and
the total requirement from 42.13 GB to roughly 15.2 GB, which fits with room to
spare. Measured, it came to 6.15 GB of data.

Per-case detail pages need one DuckDB query per case rather than an ORM join.
That is a millisecond-scale read against a columnar file with statistics, and it
is bounded by the case, so the cost is acceptable and the accessors are already
written for it.

A consumer wanting relational joins against drug or reaction rows has to work in
DuckDB rather than the ORM. That is the intended direction: those joins are
analytical, and doing them in Postgres was never going to be the fast path.

If the disk constraint disappears, this decision can be revisited by restoring
the models and re-loading from Parquet, which is a re-run rather than a
migration. The Parquet partitions are the source of truth for these datasets,
so nothing is lost by not having them in Postgres today.
