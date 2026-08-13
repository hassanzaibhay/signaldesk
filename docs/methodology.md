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

### A known weakness of the similarity criteria

Roughly 19% of blockable records carry exactly one drug and one reaction. For a
one-element set, Jaccard is either 0 or 1, so the 0.8 threshold degenerates into
an exact-match test on a single drug and a single term. Within a block of the
same sex, age and country, that will pair unrelated patients who happen to share
a common drug and a common event.

This inflates the *pair* count substantially without inflating the *record* count
to the same degree, because one record pairs with many. It also makes the choice
of survivor within a large matched clique close to arbitrary. The rate is
reported per record, not per pair, for this reason.
