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

## The figures this pass produced are provisional

Stage 2 compares raw drug name strings. One ingredient published under several
trade names, or with a dose in the string, counts as several distinct set
members, so the comparison is weaker than it looks.

The pass is therefore repeated against normalized ingredients once drug
normalization exists, that run produces the canonical duplicate table, and the
delta between the two is reported. Until then these numbers stand as a baseline
rather than as the answer.

The baseline to measure that delta against, by set size, probabilistic matches
only:

| Subset | Blockable | Matched | Rate |
|---|---|---|---|
| Sparse: one drug, one reaction | 2,417,690 | 952,693 | 39.41% |
| Richer | 9,131,468 | 701,197 | 7.68% |

Sparse records match at 5.1 times the rate of the rest, and produce 58 percent
of all matches from 21 percent of the population. On a one-element set Jaccard
is binary, so the threshold reduces to exact agreement on a single drug and a
single term, which inside a block already fixed to one sex, age and country is
not a demanding test.

Note the direction this implies for the repeat run: normalization increases
agreement between sets, so the sparse subset should match more after it, not
less. A rise there is expected and is not by itself evidence of a problem; it
has to be read against the audit rather than at face value.

### The rate is provisional until precision is measured

Because 58 percent of stage 2 matches come from the weakest-evidence subset,
nothing measured so far separates a high true duplicate rate from a high false
merge rate. The overall figure is therefore marked provisional in the committed
report and stays out of the README until a hand audit of sampled pairs gives a
false-positive rate per subset. Stage 1 is unaffected: it applies the published
rule deterministically.

### Stratum definition

The sparse stratum is defined on ingredient-set cardinality, not on raw
drug-string count: a record is sparse when its normalized ingredient set has
cardinality one and it has exactly one reaction.

This matters because the two definitions disagree. Stage 2 currently compares
raw drug strings, so a single string naming a combination product expands into
several ingredients once normalized, and the record leaves the sparse stratum
without anything about the underlying report having changed. The consequences:

- The rates measured here, 39.41 percent sparse against 7.68 percent richer,
  were computed over the raw-string stratum definition. They are not comparable
  to post-normalization figures.
- The re-run recounts stratum membership from ingredient-set cardinality rather
  than re-executing the matcher against strata assigned before normalization.
  Reporting a naive delta would credit normalization with a shift that is partly
  definitional.
- The audit sample is drawn from the post-normalization population, and the
  stratum definition is recorded alongside the seed and the commit, so the
  binomial interval attaches to a denominator that can be identified later.

Separately, stage 2 today evaluates every case, including those stage 1 has
already superseded, and discards matches on superseded records when results are
written. That is redundant computation, and it leaves the stage 2 denominator
inconsistent with its numerator: 1,663,031 cases sit in the denominator unable
to reach the numerator. The re-run restricts stage 2's population to stage 1
survivors, which removes both problems and moves the rate. The pre-run and
post-run stage 2 rates are therefore not comparable for this reason as well as
for the stratum redefinition.

### Defects carried into the re-run

Found while reconciling the published figures, all in the deduplication pass, all
fixed as part of the re-run rather than in the reporting pass that found them:

1. Stage 2's population is restricted to stage 1 survivors, as above. This is the
   change that moves the rate.
2. `stats_from_store` counts every case with a complete blocking key, omitting
   the `QUALIFY count(*) OVER (...) > 1` clause the pass applies, so its stage 2
   denominator exceeds what the pass considered by 4,646. Both paths apply the
   same filter.
3. The pass keeps no counter for probabilistic matches discarded by the
   supersession skip, so that quantity is not recoverable from stored state and
   has to be re-derived by a second pass. Add the counter.
4. The supersession skip is one-sided: it discards a match when the flagged
   record is superseded, not when the record it matched is, which leaves 98,376
   flags naming a canonical stage 1 removed. Restricting the population per item
   1 removes the case, so this is a consequence to verify gone after the re-run
   rather than a separate fix.

### Candidate rules, to be chosen after the audit and not before

1. **Corroboration requirement.** A sparse pair additionally requires agreement
   on at least one high-information field: event date at day precision,
   manufacturer sender, or weight within tolerance.
2. **Exclusion.** Sparse records are excluded from stage 2 and counted,
   symmetric with the existing exclusion for an incomplete blocking key, making
   the reported rate an explicit lower bound.

Option 2 is the conservative default if the audit shows a high false-positive
rate and option 1 does not bring it down far enough. Both are defensible; the
audit decides which, because a 10 percent error rate and a 70 percent error rate
call for different rules.

The approach a production system would take is inverse-frequency weighting,
scoring a shared rare drug as strong evidence and a shared common one as almost
none, rather than treating set members as equal. It is named here as the right
long-term answer and not adopted now, because calibrating the weights and the
score threshold needs labelled pairs that this source does not provide. The
audit is the first step towards having any.
