# 5. Stage the deduplication input on disk

Date: 2026-08-13

## Status

Accepted

## Context

Deduplication was written to load every blockable case, with its drug and
reaction sets, and then compare within blocks. That worked on the five-quarter
acceptance range of 1,050,931 cases. It failed on the full corpus of 20,534,506,
three times:

```
OutOfMemoryException: could not allocate block of size 256.0 KiB (3.7 GiB/3.7 GiB used)
OutOfMemoryException: failed to allocate data of size 32 bytes (7.4 GiB/7.4 GiB used)
```

The first failure was the query building per-case sets over 84.6 million drug
rows and 66.3 million reaction rows. Raising the memory limit from 4 GB to 8 GB
and allowing the engine to spill moved the failure from 3.7 GiB to 7.4 GiB,
which is the shape of a problem that does not have a memory-limit answer: the
working set grows with the corpus, and the next corpus is larger than this one.

Two operations were at fault, both unbounded:

* Collapsing three joined datasets into per-case sets and returning the result
  to the client, which materialises all of it at once.
* Sorting twenty million rows, each carrying two sets, to bring block members
  together.

## Decision

The deduplication input is staged to disk and read in bounded groups.

The per-case drug and reaction sets are built with `CREATE TABLE`, which streams
output to the database file rather than accumulating a result for the client.
They are joined to the cases in the same way, filtered to blocks with more than
one member, and the intermediate tables are dropped.

Blocks are then read one coarse group at a time, by (sex, country), sorted by
age within the group. One country's records sort comfortably; the whole corpus
does not. Blocks arrive in key order within a group, so the reader still yields
one block at a time and holds only that block.

The DuckDB connection also gets a temporary directory so it can spill, and
insertion-order preservation is disabled, since no query here depends on row
order.

## Consequences

The full corpus deduplicates in about an hour, comparing 705,158,714 pairs where
the naive within-block comparison would have been 221,525,158,953 - a factor of
314, on a largest block of 109,667 records. Neither the engine nor the process
exceeds its memory limit.

The staged tables cost disk in the analytics database while the pass runs. That
is the trade being made: disk is the resource this project has, and memory is
the one it does not.

A general lesson worth keeping, since the remaining corpora are larger than this
one: an operation whose working set scales with the corpus cannot be fixed by
raising a limit, and the fact that raising it moved the failure rather than
removing it is the signal to change the shape instead.
