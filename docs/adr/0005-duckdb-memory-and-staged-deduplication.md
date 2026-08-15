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

### Defects carried into the re-run - all seven closed in P03

Six were found while reconciling the published figures; the seventh was found in
P03 by running the corrected pass twice rather than by reading it. All are in the
deduplication pass, and all are fixed in P03 rather than in the pass that found
them. What each one was and how it was closed:

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
5. The survivor tie-break is not a total order. `_compare_block` keeps `left`
   when `left.fda_dt >= right.fda_dt`, which on equal dates holds in both
   directions, so the survivor is decided by argument position and therefore by
   scoring order rather than by the data. It becomes `(fda_dt, primaryid)`,
   which is antisymmetric and independent of the order pairs arrive in. See
   below for what the current ordering produced.
6. The quality report computes the surviving-record count by subtracting a
   Postgres flag count from the Parquet case total, mixing the two stores across
   a 707-row gap that ADR 0004 records. Both terms come from Postgres.
7. Canonical selection is not a choice at all. A record matching several others
   in its block overwrites its stored canonical on every match, so it keeps
   whichever comparison ran last. The pair set is unaffected - the same records
   are flagged either way - but the pointer each flagged record carries is
   decided by scoring order, which is not data. This is item 5's defect one level
   up, and it is independent of item 5: fixing the pairwise tie-break makes each
   comparison decidable without making the choice between several decided
   comparisons any less arbitrary. It becomes the strongest of the partners the
   record itself matched, on the same `(fda_dt, primaryid)` total order.

   State what that does and does not give, because the first draft of this entry
   overstated it. Selection is a **single hop over the matched neighbourhood**,
   not over the block. Jaccard is not transitive, so if `a` matches `b` and `b`
   matches `c` while `a` and `c` fall below the threshold, `a`'s canonical is `b`
   and `b`'s is `c`. **A canonical may therefore itself be flagged, and chains
   through a block still exist.** What the fix removes is the arbitrariness: the
   pointer is now a function of the data rather than of scoring order, and two
   passes over the same corpus write the same one.

   Chains are intended here rather than a further defect. A block is not an
   equivalence class - it is a set of records sharing a blocking key, related
   pairwise by a similarity that does not compose - so there is no single
   "block survivor" to point at. Resolving transitively to one would merge `a`
   into `c` on the strength of a match neither record made, which is exactly the
   over-merge direction the sparse stratum already leans in and the D2 audit
   exists to bound. The pointer stays a statement about a pair, because a pair is
   what was measured.

   **The unique-case count does merge `a` into `c`, and that is a separate
   decision.** It has to be said here, because the paragraph above reads as a
   claim that nothing in the pipeline resolves transitively, and one thing does.
   `rates.unique_cases.count` is 20,534,506 - 4,479,514 = 16,054,992: the corpus
   minus every record holding an outgoing edge. Under a chain `a -> b -> c` with
   `c` unflagged, both `a` and `b` are subtracted, so three records collapse to
   one survivor even though `a` and `c` never matched. Single hop is what is
   stored per record; transitive closure is what is counted.

   The two are consistent, and the reason is the acceptance condition itself.
   Once every chain is required to terminate at an unflagged record - zero closed
   components, zero cycles - the components of the flag graph partition the
   flagged population, and a subtraction is the only count those components
   admit: each component contributes exactly one survivor. Any count that
   credited `a` with its own survivor would have to name a record for it, and
   there is none, because `a` is flagged. So the count is not derived from the
   pointer semantics and does not follow from them; it is a second choice, made
   deliberately, that the graph be read as components where the pointer is read
   as pairs.

   **The consequence for D2, stated because it changes what the audit has to
   record.** A false-positive rate estimated over sampled pairs does not transfer
   to the unique count. One spurious edge inside a chain of length `k`
   over-merges `k - 1` records, so an error rate per pair maps onto an error in
   the count that scales with the chain the pair sits in. The D2 frame therefore
   has to carry the chain length of each sampled pair, or its interval is an
   interval on the pair rate quoted as though it bounded the count, which
   understates the count's error by whatever the mean chain length turns out to
   be. Sampling pairs uniformly and reporting a single rate is not sufficient
   here.

Closed as follows. Item 1: `stage_population` classifies every case and stages
only stage 1 survivors. Item 2: that same function builds the population for the
pass and for the report, so there is no second query to disagree with the first.
Item 3: `probabilistic_discarded_superseded`, saved and reloaded with the rest of
the run's statistics. Item 4: asserted on both sides of every written pair, over
a corpus built to contain the case, since the published fixture has no case
published at two versions in different quarters and therefore cannot exercise it.
Item 5: `choose_survivor`, ordering on `(fda_dt, primaryid)`, tested with the
arguments both ways round. Item 6: staging collapses to one row per case, so the
population stops counting the 707 republications, and every rate divides Postgres
by Postgres and names the store. Item 7: `_compare_block` carries a per-record
best-survivor key and keeps a match only when it beats the one already recorded.

Three integration tests hold item 7, and they hold different things. Two use a
fixture of four mutually-matching records sharing one `fda_dt`: two passes write
identical pairs with identical canonicals, and the highest identifier is the
canonical of all three others. Note what those two cannot see. Every record in
that fixture matches every other, so the matched neighbourhood **is** the block,
and single-hop selection and transitive resolution to a block survivor produce
the same answer. They establish determinism; they do not discriminate between the
two semantics. The third does: three records where `a` matches `b` and `b`
matches `c` but `a` does not match `c`, asserting `a`'s canonical is `b` rather
than `c`, under every order the comparison could reach them in. It is a
characterization test - it locks in the behaviour described above as
implemented, and it is the test that has to be revised, deliberately, if
transitive resolution is ever adopted.

The two fixtures separate different things, and neither covers the other. In the
path fixture each record has exactly one strictly-greater partner, so there is no
contest for the best-key guard to settle and it passes against the pre-fix
selection as well - confirmed by running the P02 comparison over it, not assumed.
The clique fixture is what fails without the guard. Determinism and single-hop
are independent properties and are tested independently.

Item 7 is worth noting as a method result and not only as a defect. Item 5 was
argued from reading the code and reading the code missed the one above it; item 7
was found by running the pass twice over records built to tie and comparing the
output. The overstatement corrected above was found the third way, by an audit
that asked what the tests would still pass if the claim were false. The
acceptance condition below is written for the same reason - verified by resolving
the pointers, not by arguing that the fixes cover it.

### What the population is, and why three classes leave it

The stage 2 rate is reported against the cases the rule can apply to: stage 1
survivors that carry a complete blocking key and a non-empty drug set. Three
classes are excluded, and all three for one reason - **they cannot reach the
numerator, so leaving them in the denominator reports a rate against a population
the rule was never applied to.**

- **Superseded by stage 1.** The published rule already withdrew them.
- **Null blocking key.** They cannot be blocked, and a null key is not evidence
  of similarity.
- **Empty drug set.** `jaccard` scores an empty set as dissimilar, so such a case
  can never match. Note the rationale: this is denominator consistency, not
  comparison safety. The comparison already does the right thing.

A case alone in its block is **not** excluded. The rule applied to it and found
nothing, which is a legitimate zero, so it stays in the denominator and is
counted separately. That count is what made the pass and the store disagree by
4,646 in P02, and it is now a named term rather than a gap.

Empty drug sets are rare today - 2 cases in 20,534,506, both inside the blockable
population, against 55 with no reaction term - but the count grows once
`not_a_drug` overrides start removing members, which is why the handling is in
place before it is needed.

### Sparse has three paths, not one

The sparse stratum is defined on ingredient-set cardinality, and after
normalization a record can arrive at cardinality one by three different routes.
They are counted separately, or the sparse delta absorbs them and is read as a
normalization effect:

- **sparse-normalized** - one resolved ingredient.
- **sparse-fallback** - one member, and it is a cleaned raw string that failed to
  normalize. Matching on it is exact string equality, the degenerate case the D2
  audit exists to bound.
- **sparse-by-exclusion** - two members became one because the other was ruled
  `not_a_drug`.

A fallback member counts toward cardinality, because the set the matcher compares
and the set the stratum is computed on must be the same object. A `not_a_drug`
member does not fall back: a non-drug string as a set member manufactures overlap
between unrelated cases.

### The delta has three components and they are not additive

Reporting a naive before-and-after would credit normalization with changes it did
not cause. Three components:

1. **Defect fixes** - the P02 baseline against a pass with all six defects fixed,
   still comparing raw drug name strings.
2. **Stratum redefinition** - that same population relabelled under
   ingredient-set cardinality. A relabelling, not a matching run.
3. **Normalization** - the intermediate pass against the ingredient-set pass.

Components 1 and 3 move the duplicate count and sum to the total delta. Component
2 moves no count at all; it changes which stratum a record belongs to and bears
on stratum-level attribution only. Anyone trying to reconcile three numbers
against one total will fail, which is why the report says this wherever the three
appear.

### Seventeen components of the flag graph have no survivor

Every flagged record carries exactly one canonical, so the flags form a
functional graph and the surviving corpus is the set of records no flag names as
a duplicate. Subtracting the flag count from the corpus size assumes every chain
of canonical pointers ends at an unflagged record.

Resolving all 4,505,344 pointers, 4,505,287 do. The remaining 57 do not: 17
connected components in which every member is flagged, each containing a cycle
of length 3, covering 38 distinct case identifiers. Those cases are removed from
the corpus outright rather than merged into a survivor, and no rate reported
anywhere says so.

The mechanism is two arbitrary choices in stage 2 meeting stage 1, not one.

Item 5 is the first. In 14 of the 17 components every member shares one
`fda_dt`, so the pairwise direction was decided by scoring order; stage 1's
direction is decided by `caseversion`. Two orderings that need not agree, in one
pointer graph, close a loop:

```
101596041 -max_caseversion-> 101596042    case 10159604 v1 -> v2
101596042 -probabilistic--> 101596061     all three fda_dt 2014-05-08
101596061 -probabilistic--> 101596041
```

Item 7 is the second, and it acts on the same graph independently. A record
matching several others kept the canonical from whichever comparison ran last,
so which of its partners the pointer named was decided by scoring order. Two
records could therefore be pointed at each other by two comparisons that never
agreed on a direction, regardless of how any individual comparison was decided,
so it can close a loop without item 5 being involved at all.

Both fed the pointer graph these components were found in, and the components
were counted before either was understood.

**Which of the 17 came from which is not recoverable.** Attributing them
requires the P02 flag table, which no longer exists, and the replay recorded in
`docs/methodology.md` establishes that re-running that code would write a
different flag set rather than reproducing this one - the components are a
sample of what the two arbitrary choices happened to produce on one pass, not a
measurement that can be re-derived and then partitioned. What is recorded here
is the mechanism and its two causes; the per-component attribution is not, and
is not going to be.

Item 1 removes the paths that run through stage 1, but a cycle confined to
stage 2 stays reachable while either arbitrary choice stands, which is why items
5 and 7 are fixes in their own right rather than consequences of item 1. The
re-run therefore verifies zero cycles and zero fully-flagged components as an
acceptance condition, rather than assuming the population change resolved it.

#### Why the corrected graph is acyclic, stated as the argument that holds

Not "because the ordering is now a total order". That would be the argument if
one ordering governed the whole graph, and it does not: stage 1 orders on
`(caseversion, primaryid)`, stage 2 on `(fda_dt, primaryid)`, and `chain_report`
reads both kinds of edge into a single graph with no filter on `method`. Two
total orders in one graph are no better than the two partial ones that closed the
P02 loops, so the argument has to come from the population instead.

Three steps, each with the code that enforces it:

1. **At most one outgoing edge per node.** Stage-1-superseded records are
   excluded from `dedup_input` by the `status = 'eligible'` join in
   `stage_population`, so they never enter a block and never acquire a stage 2
   edge; and any that somehow did have their stage 2 row dropped at write time by
   the supersession skip in `_duplicate_rows`, which is what the counter in item
   3 measures. So a node carries a stage 1 edge or a stage 2 edge, never both.
2. **Superseded records have in-degree zero.** They are not stage 1 targets - the
   target is by construction the highest version - and they cannot be a stage 2
   canonical, because a canonical is drawn from `dedup_input`, which excludes
   them. A node with no incoming edge cannot lie on a cycle.
3. **What is left is stage 2 only.** Any surviving cycle would have to run
   entirely through eligible nodes joined by stage 2 edges, and every stage 2
   edge points from lower to strictly higher `(fda_dt, primaryid)` - strictly,
   because `primaryid` is unique per staged case and same-case pairs are skipped.
   A cycle would need that strict increase to return to its start.

Two things this argument depends on that are worth naming rather than assuming.
The one-outgoing-edge invariant of step 1 is enforced at read time by
`resolve_chains`, which **raises** when a record carries two different
canonicals. It is not enforced by the schema: the constraint is
`UniqueConstraint(["primaryid", "method"])`, which permits exactly one row per
rule and therefore permits the two-edge case the argument rules out. And step 3
is about the corrected pass only - it says nothing about P02, where the stage 2
edge direction was not monotone in anything.

This is why the acceptance condition is a measurement and not a proof discharged
on paper: the argument holds given the population filter, and the pointer
resolution is what checks that the population filter did what it says.

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
