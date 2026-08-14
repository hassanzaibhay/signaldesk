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

| Rate | Measured | Population | Status |
|---|---|---|---|
| Stage 1 supersession | 2,851,499 / 20,535,213 = **13.89%** | every case | Settled |
| Stage 2 match | 1,653,845 / 11,549,158 = **14.32%** | cases with a complete blocking key | Provisional |
| Overall duplicate | 4,505,344 / 20,535,213 = **21.94%** | every case | Provisional |

**Which denominator the stage 2 rate uses.** 14.32 percent is computed against
11,549,158, every case with a complete blocking key. That is the population the
rule applies to, and it is the larger of the two candidate denominators. The pass
itself blocked 11,544,512 of those cases; the remaining 4,646 have a complete key
but are alone in their block, so nothing was ever compared against them. Against
the pass's denominator the same numerator gives 14.33 percent. The two figures
below that qualify the rate - the superseded share and the 16.74 percent - are
computed on the pass's 11,544,512, because they describe what the pass did rather
than what the rule covers.

**16,029,162 records remain once everything flagged is removed**, and that is a
record count, not a count of surviving events. Both terms come from Postgres:
20,534,506 distinct case identifiers less the 4,505,344 records the duplicate
table flags. The committed report artifact prints 16,029,869 instead, because it
subtracts the same Postgres flag count from the Parquet case total of
20,535,213. Parquet holds one row per publication and Postgres one per case, a
707-row gap recorded in ADR 0004, so the two cannot appear in one subtraction.
Both terms come from Postgres once the report generator is changed with the rest
of the deduplication fixes; the change is recorded in ADR 0005, and until it
lands the figure to quote is the one stated here rather than the one in the
artifact.

The count is also provisional, because stage 2 is. Stage 1 at 13.89 percent is
the only settled rate here, and there is no duplicate rate for this corpus that
can be quoted on its own yet. One becomes reportable when the audit bounds the
stage 2 false-positive rate.

**38 case identifiers survive nowhere, which is a defect and not a property of
the method.** Subtracting the flagged records treats every one of them as having
a surviving representative elsewhere in the corpus. Resolving the canonical
pointers shows 17 components in which every member is flagged and no pointer
reaches an unflagged record: 57 records, 38 distinct case identifiers, removed
outright rather than merged into a survivor. That is 57 records in 16,029,162,
about 3.6 in a million, and the size is not the point - the events are absent
from the corpus, and no rate computed here says so. The cause is a survivor
tie-break that is not a total order, recorded in ADR 0005 and fixed in the
re-run, which verifies afterwards that no such component remains.

**Why the stage 2 rate is a lower bound.** Two mechanisms hold it down, and
neither is a property of the data:

- 8,986,055 cases, 43.76 percent of the corpus, are excluded from stage 2
  entirely because sex, age or country is missing and they cannot be blocked.
- 1,663,031 cases, 14.41 percent of the stage 2 denominator, were already
  superseded by stage 1. They sit in the denominator but cannot reach the
  numerator, because a match on a record that stage 1 already resolved is
  discarded when the results are written. Excluding them would put the rate at
  16.74 percent rather than 14.33.

A third, smaller effect is the 4,646 single-member blocks noted above: they
carry a complete key, so they are in the published denominator, but nothing was
ever compared against them.

**On the arithmetic.** The three counts add exactly: 2,851,499 + 1,653,845 =
4,505,344. That is not because the populations are disjoint, since both stages
range over the whole corpus and 1,663,031 cases appear in both. They add because
a probabilistic match is discarded at write time when its record is already
recorded as superseded, so no case is counted under both rules.

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
