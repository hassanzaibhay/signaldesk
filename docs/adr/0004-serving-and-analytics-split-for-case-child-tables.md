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

## Consequences

Projected Postgres for the full corpus falls from 31.18 GB to about 4.2 GB, and
the total requirement from 42.13 GB to roughly 15.2 GB, which fits with room to
spare.

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
