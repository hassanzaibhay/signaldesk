# Methodology

Definitions that downstream numbers depend on. Each is pinned here and in the
code so the two cannot drift apart.

Disproportionality statistics and their validation arrive with the statistics
layer; this document currently covers the ingest and deduplication decisions
that bound everything computed later.

## Unit of analysis

The **case**, not the report and not the drug row. A case that names four drugs
and three reactions contributes one record, not twelve. Getting this wrong
inflates every statistic computed from it.

The case is identified by `primaryid`. That identifier is unique within a
quarter but **not across the corpus**: the source republishes a case, unchanged
and at the same version, in consecutive quarters. A republished case therefore
replaces its earlier copy on load rather than being inserted twice.

## Seriousness

Neither field exists in the source. Both are derived from the outcome table, and
the codes themselves are kept on the case as evidence.

* **`serious_death`** - the case has an outcome row with code `DE`.
* **`serious`** - the case has **any** outcome row. Every code this source
  publishes is a seriousness criterion, so presence is the definition and no
  subset is enumerated.
* **`outcome_codes`** - the distinct codes the two booleans were derived from.

**A case with no outcome rows is treated as not serious, rather than as
unknown.** That is an assumption about the source, not something it states. It is
recorded here because it is exactly the kind of thing a reviewer should
challenge, and because a serious-only sensitivity analysis needs to know that
"not serious" includes "no outcome information was supplied".

## Partial dates

The source publishes dates at three precisions: a full date, a year and month,
or a year alone. Each date is stored three times: the raw string exactly as
published, a parsed date, and a precision flag of `day`, `month`, `year` or
`missing`.

Year and month precision are materialised as the first day of the period so the
column has a single type. **Nothing may compute a duration without consulting the
precision flag.** A time-to-onset derived from a year-only value padded to
January the first is a fabricated number, not an approximate one. Any consumer
measuring an interval filters on `precision == "day"` at both ends and reports
how many records that excluded.

Impossible calendar dates such as `20130231` parse to null while keeping their
stated precision, so they are visible as a data quality count rather than as
silently missing dates.

## Units

Age is converted to years and weight to kilograms, from the unit code the source
supplies alongside each value.

| Age code | Years | Weight code | Kilograms |
|---|---|---|---|
| `DEC` | 10 | `KG` | 1 |
| `YR` | 1 | `LBS` | 0.45359237 |
| `MON` | 1/12 | `GMS` | 0.001 |
| `WK` | 1/52.1775 | | |
| `DY` | 1/365.25 | | |
| `HR` | 1/8766 | | |
| `MIN` | 1/525960 | | |
| `SEC` | 1/31557600 | | |

A value with no unit code is **not** assumed to be years: an unlabelled number
could be months, and guessing would distort the age distribution silently.

Ages outside 0 to 120 years and weights outside 0 to 650 kg become null and are
counted. One row in 2013Q1 carries `YEARS` in the weight unit column; it is
counted separately as a corrupt unit rather than folded into ordinary missing
weights, because it means something different.

An **unrecognised** unit code raises rather than nulling. A new code upstream is
a schema event, and a silently nulled age disappears from every age-stratified
analysis without trace.

## Country

The occurrence country is canonical. Where it is absent the reporter country is
used instead, and `country_source` records which was taken. The fallback recovers
a substantial number of records: in 2013Q1 it supplies the country for 39,577 of
223,016 cases.

The literal value `COUNTRY NOT SPECIFIED` is normalized to null. This is not a
mapping of country names, which stays out of ingest; it is recognising an
explicit statement of absence. Left as a value it would form a single large
deduplication block whose members share only the fact that their country is
unknown.

## Deduplication

Two stages, measured separately.

### Stage 1: the published rule

Keep the highest case version per case identifier, and remove case identifiers on
the quarter's deleted list. Version resolution runs both within a quarter during
ingest and across the corpus afterwards, because a case first reported in one
quarter and updated in a later one appears in both.

### Stage 2: probabilistic matching

The same event is routinely re-reported under a **different** case identifier by
a different reporter. Records are blocked on (sex, age rounded to years,
country), and a pair within a block is judged a duplicate when all of:

* drug-set Jaccard >= 0.8
* reaction-set Jaccard >= 0.8
* same sex, same country
* age difference <= 1 year

The most recently received report survives; the others are recorded in
`faers_duplicate` with the similarity scores that produced the judgement.

Comparison inside a block uses size and prefix filtering rather than all pairs.
Both filters are exact - they discard only pairs that provably cannot reach the
threshold - so the result equals the naive comparison while doing far less work.

### What the duplicate rate does not include

**A record missing sex, age, or country is excluded from stage 2 entirely.** It
is not placed in a null block and not matched on the remaining keys. A null
blocking key is not evidence of similarity, and relaxing the key to admit such
records would manufacture merges between cases whose only commonality is absent
data.

The consequence, which must travel with the number: **the reported duplicate rate
is a lower bound.** Records too sparse to block reliably cannot be matched. The
excluded count and share are reported alongside the rate rather than buried.

### What is settled and what is not

Stage 1 is settled. Version supersession applies the source's own published rule
deterministically: keep the highest version per case identifier. There is no
estimate in it and nothing to validate.

**Stage 2 is provisional, and its output does not go in the README yet.** It is a
probabilistic judgement whose false-positive rate has not been measured. The
reason for caution is specific rather than general: 58 percent of its matches
come from records carrying one drug and one reaction, where the similarity test
reduces to exact agreement on two common values. Nothing measured so far
distinguishes a genuinely high duplicate rate from a high false merge rate, and
published probabilistic deduplication work on this source generally reports
lower figures than 21.94 percent.

A number without evidence behind it does not ship, so the overall rate carries
that status in the committed report until the audit below is done.

### Measuring the false-positive rate

This source has no gold standard for duplicates, so precision has to be
estimated directly from a hand audit rather than derived.

The procedure, in the order it must happen:

1. Draw a stratified random sample of flagged pairs: pairs from the sparse
   subset and pairs from the richer subset, sampled separately because their
   error rates are expected to differ by a wide margin. Three things are
   recorded as the sample's provenance, not the seed alone: the seed, the
   stratum definition in force, and the commit the population was drawn at. A
   seed identifies a draw only if what was drawn from is also pinned, and the
   stratum definition changes with normalization.
2. Emit one row per pair carrying everything a person needs to judge it side by
   side: both case identifiers, receipt and event dates with their precision
   flags, sex, age, weight, country and which column supplied it, the full drug
   and reaction lists, outcome codes, reporter occupation, and manufacturer
   sender. The verdict field is left empty.
3. Hassan annotates against a written guideline. The labels are not generated,
   in line with the rule that ground truth is human-curated.
4. Report the false-positive rate per subset with a binomial confidence
   interval, and the correction it implies for the overall duplicate rate.

**Flags whose partner stage 1 removed must be drawable.** The write-time skip
that keeps a case from being counted under both rules is one-sided: it discards a
probabilistic match when the flagged record is superseded, not when the record it
matched is. 98,376 stage 2 flags, 5.95 percent of the stage 2 numerator, name a
canonical that stage 1 superseded. For each of them the pass removed a record
without establishing that it duplicates anything left in the population.

**Unbacked removals are at least 9,609 and at most 98,376**, 0.58 to 5.95
percent of the stage 2 numerator:

- 9,609 are certain. The version stage 1 kept could never have been compared
  against the flagged record at all - 7,202 because it blocks elsewhere, 2,407
  because it has no complete blocking key.
- The remaining 88,767 are unknown. The kept version was in the same block, so
  the comparison was available, but the pass stores one canonical per record and
  keeps no count of a record's other matches, so whether the two ever matched is
  not recoverable from stored state.

Availability is not a match, and unknown is not backed, so the upper end stands
until something resolves it. The D2 audit is what resolves it: an annotator
comparing a flagged record against the version stage 1 kept answers exactly the
question the stored state cannot. That means these pairs have to be drawable in
the sample rather than treated as settled by the skip, and they are in the
over-merge direction the audit exists to bound.

The tooling is built in the drug normalization prompt, alongside the stage 2
re-run, so the audit and the re-run inform each other.

### Rates are reported against the population each rule applies to

The two stages have different denominators, and sharing one overstates the
result. Version supersession is checked on every case. Probabilistic matching is
only attempted on cases with a complete blocking key, a strict subset.

The figures below are the P02 pass, and they are **superseded**. P03 closed the six
defects ADR 0005 carried forward and re-ran stage 2 over the corrected population;
what that changed is recorded under "What the defect fixes moved" further down, and
the current artifact is `evals/history/ingest_faers_20260814T142156Z.json`. The P02
table is kept because the delta is reported against it.

**Every P02 figure derived from the stage 2 flag set is one sample, not a
measurement.** This is stronger than provisional and it is established rather than
suspected. Replaying P02's stage 2 from commit `7b7d499` against the same corpus
reproduces every pass-level figure exactly - 11,544,512 considered, 705,158,714
comparisons scored against 221,525,158,953 naive, 19,984 blocks, largest 109,667 -
and then writes a **different flag set**: 1,653,867 stage 2 flags where the table
holds 1,653,845, and 97,930 with a stage-1-superseded canonical where the audit
found 98,376.

The cause is the tie-break. `left.fda_dt >= right.fda_dt` holds in both directions
on equal dates, so which record of a tied pair was flagged fell out of the order
the comparison happened to reach them in. The comparison set is deterministic; the
survivor selection was not.

So these carry that status: **14.32 percent, 21.94 percent, the 98,376 flags
naming a superseded canonical, the 17 closed components, the 57 records and the 38
case identifiers that survived nowhere.** They remain the right thing to report as
what the P02 pass produced, and the delta below is correctly computed against them,
but re-running that code would not produce them again. It also retires the question
of why the P02 baselines cannot simply be regenerated.

**Stage 1 is not affected.** Version supersession is deterministic SQL, and the
replay returned 2,851,499 superseded records, exactly the figure the table holds.
13.89 percent is reproducible.

P03's pass is checked for the same property directly rather than assumed to have
it: two passes over a fixture built with `fda_dt` ties write identical flag sets,
canonicals included.

| Rate | Measured | Population | Status |
|---|---|---|---|
| Stage 1 supersession | 2,851,499 / 20,535,213 = **13.89%** | every case | Settled |
| Stage 2 match | 1,653,845 / 11,549,158 = **14.32%** | cases with a complete blocking key | Superseded by P03 |
| Overall duplicate | 4,505,344 / 20,535,213 = **21.94%** | every case | Superseded by P03 |

The three paragraphs that follow describe the P02 pass and are kept because the
delta is reported against it. For what the corrected pass measures, skip to "What
the defect fixes moved".

**Which denominator the stage 2 rate uses.** 14.32 percent is computed against
11,549,158, every case with a complete blocking key. That is the population the
rule applies to, and it is the larger of the two candidate denominators. The pass
itself blocked 11,544,512 of those cases; the remaining 4,646 have a complete key
but are alone in their block, so nothing was ever compared against them. Against
the pass's denominator the same numerator gives 14.33 percent. The two figures
below that qualify the rate - the superseded share and the 16.74 percent - are
computed on the pass's 11,544,512, because they describe what the pass did rather
than what the rule covers.

Under P02 this paragraph carried a surviving-record count of 16,029,162 and an
instruction to quote it rather than the committed artifact, because the artifact
divided a Postgres flag count by a Parquet publication total. **Defect 6 is
closed, so that instruction is withdrawn and the artifact is authoritative
again.** The current figure is **16,054,992 surviving records**, both terms from
Postgres, in `evals/history/ingest_faers_20260814T142156Z.json`. Every rate in
that file names its store, and the Parquet total appears beside it for
reconciliation rather than inside a subtraction.

The count is still provisional, because stage 2 is. Stage 1 at 13.89 percent is
the only settled rate here, and there is no duplicate rate for this corpus that
can be quoted on its own yet. One becomes reportable when the audit bounds the
stage 2 false-positive rate.

**38 case identifiers survived nowhere under P02, which was a defect and not a
property of the method.** Subtracting the flagged records treats every one of them
as having a surviving representative elsewhere in the corpus. Resolving the
canonical pointers showed 17 components in which every member was flagged and no
pointer reached an unflagged record: 57 records, 38 distinct case identifiers,
removed outright rather than merged into a survivor. That was 57 records in
16,029,162, about 3.6 in a million, and the size was not the point - the events
were absent from the corpus, and no rate computed there said so. Two arbitrary
choices fed that graph - a survivor tie-break that is not a total order, and a
canonical kept from whichever comparison ran last - and which component came from
which is not recoverable; ADR 0005 records both and why. **P03 closed both, and
the corrected pass resolves all 4,479,514 pointers with zero closed components
and zero cycles.**

**Why the stage 2 rate is a lower bound.** Under P02 two mechanisms held it down,
neither a property of the data:

- 8,986,055 cases, 43.76 percent of the corpus, were excluded from stage 2
  entirely because sex, age or country is missing and they cannot be blocked.
- 1,663,031 cases, 14.41 percent of the stage 2 denominator, were already
  superseded by stage 1. They sat in the denominator but could not reach the
  numerator, because a match on a record that stage 1 already resolved is
  discarded when the results are written. Excluding them would put the rate at
  16.74 percent rather than 14.33.

A third, smaller effect was the 4,646 single-member blocks noted above: they
carry a complete key, so they were in the published denominator, but nothing was
ever compared against them.

The corrected pass removes the second mechanism entirely - superseded cases leave
the population - and keeps the other two visible as named counts rather than as
qualifiers in prose: 7,797,472 excluded for a null key, and 4,906 alone in their
block and still inside the denominator. The rate remains a lower bound for the
null-key reason alone.

**On the arithmetic.** The three counts add exactly: 2,851,499 + 1,653,845 =
4,505,344. That is not because the populations are disjoint, since both stages
range over the whole corpus and 1,663,031 cases appear in both. They add because
a probabilistic match is discarded at write time when its record is already
recorded as superseded, so no case is counted under both rules.

### What the defect fixes moved

P03 re-ran stage 2 with all six defects closed and still comparing raw drug name
strings, so this delta is the defect fixes alone. Stratum redefinition and
normalization are separate components, measured separately, and not additive with
this one: redefinition moves no count at all, it only changes which stratum a
record belongs to.

| Measure | P02 | After the fixes | Change |
|---|---|---|---|
| Records flagged | 4,505,344 | 4,479,514 | -25,830 |
| Stage 2 flags | 1,653,845 | 1,628,015 | -25,830 |
| Stage 2 population, per case | 11,548,620 | 9,885,533 | -1,663,087 |
| Stage 2 rate | 14.32% | **16.47%** | +2.15pp |
| Cases with no surviving representative | 38 | **0** | -38 |

**The population row is a recount, not the published figure.** P02 published
11,549,158, which counts publications. Differencing that against a per-case count
would state a delta between two different measurements, which is the error defect 6
exists to remove. Recounted per case in Postgres, P02's population is 11,548,620,
and the change reconciles with no residual: less 1,663,085 superseded by stage 1,
less the 2 cases with an empty drug set, gives exactly 9,885,533. The 538 between
the two P02 figures is the republication gap on cases carrying a complete key.

**The rate moved for a different reason than predicted.** The prediction above was
16.74 percent, computed as 1,653,845 / 9,881,481. The realised denominator is
9,885,533, only 4,052 away, so holding the P02 numerator against it still gives
16.73 percent. Of the 0.268 point gap between prediction and measurement, 0.261
points is the numerator falling to 1,628,015 and 0.007 points is the denominator.
The population change was predicted accurately; what was not predicted is that
25,830 matches would stop being made.

**Chain verification passes.** Resolving all 4,479,514 canonical pointers gives
zero closed components and zero cycles, so every flagged record now ends at a
record nothing flagged and the surviving-count subtraction is sound. The 17
components and 38 orphaned cases above are gone.

### A known weakness of the similarity criteria, measured

About 21% of blockable records carry exactly one drug and one reaction. For a
one-element set Jaccard is either 0 or 1, so the 0.8 threshold degenerates into
an exact-match test on a single drug and a single term.

That sounds strict, and it is not. Within a block already fixed to one sex, one
rounded age and one country, agreement on a single common drug and a single
common event happens readily by chance:

| Subset | Blockable | Matched | Rate |
|---|---|---|---|
| Sparse: one drug, one reaction | 2,417,690 | 952,693 | **39.41%** |
| Richer | 9,131,468 | 701,197 | **7.68%** |

Sparse records match at 5.1 times the rate of richer ones. They are 21% of the
blockable population and produce 58% of all probabilistic matches. The direction
of the error on these records is over-merging, not under-merging.

The threshold is deliberately left alone. Relaxing it for singletons would make
any two records sharing one drug, one reaction, sex, age and country a
duplicate, which is worse. Tightening it globally would discard real matches
among richer records, which are behaving reasonably.

The lever is minimum evidence rather than threshold, but which rule to apply
depends on how wrong the sparse matches actually are, and that is what the audit
measures. Two candidates, neither adopted before the measurement:

1. **Corroboration.** A sparse pair additionally has to agree on at least one
   high-information field: event date at day precision, manufacturer sender, or
   weight within a tolerance. Keeps sparse records in scope and demands one
   piece of evidence that is unlikely to coincide.
2. **Exclusion.** Sparse records are removed from stage 2 and counted, exactly
   as records with an incomplete blocking key already are, making the reported
   rate an explicit lower bound over a smaller population.

Option 2 is the conservative default if the audit shows a high false-positive
rate and corroboration does not bring it down.

The textbook approach is inverse-frequency weighting: a shared rare drug is
strong evidence and a shared common one is nearly none, so terms are weighted by
how often they occur rather than counted equally. That is what a production
pharmacovigilance system does, and it would handle the sparse case naturally.
It is not proposed here for an honest reason: calibrating the weights, and
choosing the threshold on the weighted score, needs labelled pairs, and this
source provides none. It is worth revisiting once the audit has produced some.

**These figures are provisional.** Stage 2 compares raw drug name strings, so one
ingredient under several trade names counts as several set members. The pass is
repeated against normalized ingredients once drug normalization exists, and the
delta is reported. Note the expected direction: normalization increases
agreement, so the sparse subset should match *more*, not less. If it does, the
lever is a minimum-evidence rule - requiring more than one distinct drug or
reaction before a probabilistic match is allowed - rather than a threshold
change.

## Disproportionality

Four estimators over the same 2x2 table, per (drug, MedDRA PT) pair:

```
            PT present    PT absent
drug            a             b
not drug        c             d
```

The unit is the deduplicated case, and `a + b + c + d` is the whole
deduplicated population for every row. That identity is asserted on every row
the builder emits. If a marginal were counted over one population and the
background over another, every cell would still be non-negative and every
estimator would still return a number; the sum is the only thing that shows it.

A case naming three drugs and two terms contributes six pairs. It is counted
once in cell `a` of each of those six, and once in the background of every pair
in the table. That is what the table means, not double counting.

### What was checked against what

Every estimator is validated against an unmodified published implementation,
not against a table worked out alongside the code. The generator script and its
output are committed under `tests/fixtures/estimators/`.

| Quantity | Reference | Agreement |
|---|---|---|
| ROR, and both 95% limits | `epitools::oddsratio.wald` 0.5.10.1 | 1e-8 |
| PRR, and both 95% limits | `epitools::riskratio.wald` 0.5.10.1 | 1e-8 |
| chi-squared, Yates corrected | R 4.4.1 `stats::chisq.test` | 1e-8 |
| IC posterior mean and variance | `PhViD::BCPNN` 1.0.8 | 1e-9 |
| IC 2.5th percentile | `PhViD::BCPNN` 1.0.8 | 1e-9 |
| MGPS likelihood, weighted | `openEBGM::negLLsquash` 0.9.1 | 1e-10 |
| MGPS likelihood, unweighted | `openEBGM::negLL` 0.9.1 | 1e-10 |
| MGPS hyperparameter fit | `openEBGM::autoHyper` 0.9.1 | 1e-3 |
| Qn, EBGM, EBGM05 | `openEBGM` 0.9.1 | 1e-6 |

### The minimum cell count is a flag, not a filter

`a >= 3` decides which pairs enter the headline counts. It does not decide which
pairs are computed or written. Two independent reasons:

* a pair filtered out here disappears from the denominator of every rate
  reported downstream, with nothing left to say it existed;
* MGPS fits its prior on the distribution of counts across the whole table.
  Handing it only pairs with three or more cases removes exactly the low-count
  mass the first mixture component exists to describe. Measured on the openEBGM
  CAERS data, that truncation moves the second component's shape from 2.15 to
  316 and the mixture weight from 0.07 to 0.75. The optimizer converges and
  reports success; the prior is simply no longer the one the data implies.

### Haldane-Anscombe: where it applies, and where it must not

Add 0.5 to every cell of a row that has an empty cell, so the ratio estimators
stay finite. Applied per row and never unconditionally: applied to every row it
would pull the whole table toward the null by an amount that grows as counts
shrink. The `corrected` column records which rows were adjusted.

It reaches the ROR and PRR point estimates and their intervals. It does **not**
reach:

* **the chi-squared**, because the Yates continuity correction is already the
  small-cell adjustment for that statistic. Applying both shrinks the same
  deviation twice and depresses chi2 exactly where the `chi2 >= 4` threshold
  decides;
* **BCPNN and MGPS**, because both are Bayesian and already shrink small counts.
  A pseudo-count on top would shrink twice.

### BCPNN: the prior, and what is reported

Bate et al. (1998). Independent Beta priors on the three probabilities the
information component is built from, with conjugate posteriors:

| Quantity | Posterior |
|---|---|
| `p(drug)` | `Beta(1 + n1., 1 + N - n1.)` |
| `p(event)` | `Beta(1 + n.1, 1 + N - n.1)` |
| `p(drug, event)` | `Beta(1 + a, g - 1 + N - a)` |

A uniform prior on each marginal, worth one prior case each way. `g` is not
free: it is fixed by requiring the prior to be centred on independence,
`E[p(drug, event)] = E[p(drug)] E[p(event)]`, giving
`g = (N + 2)^2 / ((n1. + 1)(n.1 + 1))`.

`IC` is a sum of three independent log-Beta variables, so with
`E[ln X] = psi(u) - psi(u+v)` and `Var[ln X] = psi'(u) - psi'(u+v)` for
`X ~ Beta(u, v)`, the mean divides by `ln 2` once and the three variances add
and divide by `ln 2` squared. The derivation was checked against
`PhViD::BCPNN`, whose `r2b = N - n11 - 1 + (2+N)^2/(q1*p1)` is `g - 1 + N - a`
written out.

**The reported `IC` is the posterior mean.** ARCHITECTURE section 8.2
originally specified the closed form `log2(N*a / ((a+b)(a+c)))` with
`IC025 = IC - 2*sqrt(V)`, and was amended on 2026-08-15. That pairing put a
shrunk variance under an unshrunk point estimate, which makes `IC025` least
conservative exactly where the evidence is thinnest. `g` scales as `1/E`, so
what drives the divergence is the expected count, not the observed one:

| `a` | `E` | posterior mean | observed/expected | difference |
|---|---|---|---|---|
| 5000 | 577.5 | 3.112 | 3.114 | 0.002 |
| 25 | 20.5 | 0.245 | 0.286 | 0.041 |
| 100 | 0.400 | 6.169 | 7.966 | 1.797 |
| 4 | 0.00024 | 2.268 | 14.025 | 11.757 |

On the last row the amended `IC025` is 0.54 where the previous definition gave
12.30, against an exact posterior 2.5th percentile of 0.58. The remaining 0.03
is the `2` against `1.96` multiplier and nothing else.

The unshrunk closed form is kept as `ic_observed_expected` so the divergence
stays visible on every row. It is negative infinity at `a = 0`, where the
posterior mean is finite; the builder never emits such a pair.

`IC` rises with `a` only while `a` stays small beside `b` and `c`. The
derivative of the closed form is `1/N + 1/a - 1/(a+b) - 1/(a+c)`, and `a`
appears in `N` and in both marginals, so once `a` rivals its margins a further
co-report enlarges the expectation faster than the observation.
`a=2,b=1,c=1,d=2` gives 0.415 and `a=3,b=1,c=1,d=2` gives 0.392. A large
background does not rescue it: `a=1,b=1,c=1,d=10^7` and `a=2,b=1,c=1,d=10^7`
decrease too. This never arises on FAERS, where both marginals are in the
millions.

### MGPS

DuMouchel (1999), with the estimation refinements of DuMouchel and Pregibon
(2001). `N ~ Poisson(E * lambda)` with `lambda` drawn from a two-component gamma
mixture fitted over the whole table by maximum likelihood.

Only pairs reported at least once exist in the table, so the likelihood
conditions on `N >= 1`. Each component is truncated separately and the truncated
components are then mixed, which is the ordering in `openEBGM::negLL` and is
what the published estimates are conditioned on.

**Data squashing.** The full FAERS table is 8,583,614 pairs and a direct fit
does not finish in usable time. Squashing groups pairs that differ only slightly
in `(N, E)`: within a count stratum, the ten pairs with the largest expected
counts are carried individually because they hold the tail information the two
components differ in, and the rest are sorted by `E` and binned in groups of
300, each bin contributing its mean `E` with a weight equal to the number of
pairs it stands for.

A stratum is squashed only if it would still yield at least 50 bins. That guard
is the whole safety of the procedure and it was measured, not assumed. On the
openEBGM CAERS table:

| Squashed | Rows fitted | alpha1 | Largest relative error |
|---|---|---|---|
| N=1 only, the 16,040-pair stratum | 1,213 of 17,189 | 2.9407 | 0.29% |
| also N=2, a 710-pair stratum | 516 | 22.81 | 676% |
| every stratum, unguarded | 197 | 8.02 | 469% |

Squashing a large stratum is very nearly lossless. Squashing a small one
destroys the fit while still converging and reporting success.

**Failure.** A fit that does not converge raises. It never falls back to the raw
observed-to-expected ratio and never returns NaN. A fit whose optimum sits on a
parameter bound also raises by default, because a pinned parameter means the
likelihood wanted to leave the feasible region and the reported value is an
artefact of the box. A caller may accept such a fit explicitly, and the run then
records which parameters are pinned and marks every EBGM and EBGM05 derived from
it provisional.

A boundary fit stops being provisional only when a profile likelihood has settled
it: whether the bound is the optimum, and whether the scores move across the
profile. That ruling is an explicit input, never inferred from the fit, because a
fit cannot conclude anything about its own boundary from the inside. What the
ruling changes is what is reported; which parameters were pinned stays recorded
either way.

### The MGPS boundary on this corpus, and why EBGM is reported anyway

On the full corpus the fit reaches the lower bound on `alpha1`. That is refused
by default, so two questions had to be answered before any EBGM number could be
written. Is the bound the genuine optimum for this data shape, or a defective
likelihood? And if it is the optimum, do the reported scores depend on where on
the profile one stands?

Both were answered by measurement, and the answers are below: the bound is the
optimum, and the scores do not move where the likelihood has support.

Settled by profile likelihood, not by refitting from more start points. `alpha1`
was fixed on a grid and the remaining four parameters refitted at each point,
over 497,975 squashed rows standing for 8,583,614 pairs. The grid is swept
forward and then backward, each point offering its solved vector as an extra
start to the next; both sweeps are kept, because where they disagree the
likelihood has more than one basin at that point.

The reading is one comparison: the likelihood at the bound against the best
value anywhere on the grid. Monotonicity is reported next to it as an
observation about shape and settles nothing on its own, since a curve can wander
far from the bound while the bound is still the optimum.

| `alpha1` | negative log-likelihood | excess over the minimum | sweeps agree |
|---|---|---|---|
| 1e-06 | 16,580,220.1779 | 0 | yes |
| 1e-05 | 16,580,224.5763 | 4 | yes |
| 3e-05 | 16,580,234.3505 | 14 | yes |
| 1e-04 | 16,580,268.5594 | 48 | yes |
| 3e-04 | 16,580,366.3079 | 146 | **no** |
| 1e-03 | 16,580,708.1533 | 488 | yes |
| 3e-03 | 16,581,683.5304 | 1,463 | yes |
| 1e-02 | 16,585,080.7681 | 4,861 | yes |
| 0.02 | 16,589,887.7051 | 9,668 | yes |
| 0.05 | 16,603,958.1053 | 23,738 | **no** |
| 0.1 | 16,626,118.6817 | 45,899 | yes |
| 0.2 | 16,665,094.0038 | 84,874 | yes |
| 0.35 | 16,710,405.9863 | 130,186 | yes |
| 0.5 | 16,743,082.5167 | 162,862 | yes |
| 0.75 | 16,781,297.9905 | 201,078 | yes |
| 1.0 | 16,810,479.9780 | 230,260 | yes |
| 1.5 | 16,856,587.7199 | 276,368 | yes |
| 2.0 | 16,893,696.1952 | 313,476 | yes |
| 3.0 | 16,951,722.5586 | 371,502 | yes |

The bound is 4.40 nats below the best grid point, at `alpha1 = 1e-05`, and the
curve rises monotonically away from it. **The bound is the optimum, not a
defective likelihood.** A zero-truncated negative binomial tends to the
logarithmic distribution as its shape goes to zero, and that is a legitimate
limit for a table where 68 percent of pairs are reported exactly once.

The two sweeps disagree at `alpha1 = 3e-04` and `alpha1 = 0.05`, and each
direction loses one of them: forward reaches 16,580,366.3079 at 3e-04 against
backward's 16,580,366.8990, and backward reaches 16,603,958.1053 at 0.05 against
forward's 16,604,082.9176. Recorded rather than merged away, because the
disagreement is the evidence that those grid points carry more than one basin.

An earlier single-direction profile put `alpha1 = 0.05` at 16,633,494.65, above
its neighbour at 0.1, and that irregularity was recorded here as a nested-refit
local optimum. Continuation confirms the diagnosis and removes it: the point
now sits 29,536 nats lower, at 16,603,958.11, and below 0.1 as the shape
requires. The earlier figure described the search, not the likelihood.

**The remaining question was whether the scores depend on the answer.**
Rescoring all 8,583,614 pairs at points across the profile, over the 2,785,896
pairs at or above the minimum cell count:

| `alpha1` | flagged, EBGM05 > 2 | median | 90th pct | 99th pct |
|---|---|---|---|---|
| 1e-06 (bound) | 1,133,093 | 1.3804 | 10.3960 | 499.5659 |
| 1e-05 (grid argmin) | 1,133,100 | 1.3805 | 11.1331 | 506.3270 |
| 1e-03 | 1,134,008 | 1.3810 | 17.4020 | 507.2114 |
| 1e-02 | 1,141,400 | 1.3867 | 28.6227 | 507.3675 |
| 0.1 | 1,174,917 | 1.4237 | 33.1268 | 510.0281 |
| 1.0 | 1,379,260 | 1.9547 | 41.2583 | 531.6345 |
| 3.0 | 1,301,053 | 1.6533 | 55.7130 | 501.5227 |

**They do not.** The bound and the grid argmin - the two points the comparison
turns on, and the region where the likelihood has support - differ by **7 flagged
pairs out of 1,133,093**, 6.2 parts per million. That is the sensitivity figure.

The full-profile spread is 21.7 percent, from 1,133,093 to 1,379,260, computed as
`(flagged_maximum - flagged_minimum) / flagged_minimum`. It is reported as
context, not as a sensitivity result: it is driven by `alpha1 = 1`, which the
likelihood puts **230,259.80 nats** worse than the bound. A spread measured
across grid points the data excludes does not describe uncertainty in the
reported count.

**So the boundary reading is non-load-bearing.** Whether the logarithmic limit is
read as a prior or as a degeneracy does not change the count to within seven
pairs, so that question does not have to be settled for the number to be
reported. EBGM and EBGM05 are measurements and are quotable, and `flag_all_four`
is computed:

| | PS+SS primary | PS sensitivity |
|---|---|---|
| ROR | 1,693,733 | 769,906 |
| PRR | 1,623,081 | 726,832 |
| BCPNN | 1,393,815 | 603,100 |
| MGPS | 1,133,092 | 360,724 |
| `flag_ror_prr_bcpnn` | 1,393,815 | 603,100 |
| **`flag_all_four`** | **1,113,770** | **360,303** |

`flag_all_four` is the conservative definition of ARCHITECTURE section 8.2, all
four thresholds met. It sits below the MGPS count because MGPS flags 19,322 pairs
(PS+SS) and 421 (PS) that BCPNN does not; BCPNN is otherwise a strict subset of
both ROR and PRR, which is why `flag_ror_prr_bcpnn` equals the BCPNN count.

The two runs were built while MGPS was withheld and wrote `flag_all_four` as
null. The counts above are recomputed from what those runs persisted -
`flag_ror_prr_bcpnn & flag_mgps` per row, headline over pairs at or above the
minimum - rather than by rebuilding, so each run stays immutable and keeps the
`run_id` the diagnostic names. Their parquet column is still null; the next build
writes it natively.

Every figure above comes from `signaldesk signals mgps-diagnostic` and
`signaldesk signals artifact --mgps-adjudicated`, recorded in
`evals/history/signals_20260818T051858Z.json`.

### Reference-set validation

Not yet performed. `evals/reference_sets/` is empty: the Harpaz, OMOP and EU-ADR
standards and the outcome-to-MedDRA-PT map are curated by hand and are never
generated by this project. `signaldesk evals run signals` refuses and names each
required file and its schema. No AUROC, sensitivity or specificity appears
anywhere until those files exist.
