# 8. What confidence in a drug match should mean

Date: 2026-08-14

## Status

Proposed. Candidate signals and the evidence that would separate them. No signal
is chosen here.

## Context

The plan for P03 was a confidence floor on the RxNav approximate score: matches
above it are accepted, matches below are `NULL` and counted. Ten hand probes,
committed as `evals/history/rxnav_probe_20260814T060235Z.json` before any
matching code existed, show that a floor on that score cannot do the job.

The score is not a normalized confidence. Across ten terms it ranges from 3.31 to
42.40 and it tracks the length of the query rather than the quality of the match:

| Query | Score | Resolved ingredient | Correct |
|---|---|---|---|
| `PFIZER-BIONTECH COVID-19 VACCINE` | 42.40 | SARS-CoV-2 vaccine, mRNA | yes |
| `UNSPECIFIED INGREDIENT` | 10.18 | caviar preparation | **no** |
| `ACETAMINOPHEN.` | 9.65 | acetaminophen | yes |
| `ACETYLSALICYLSAEURE` | 3.31 | aspirin | yes |

Any floor placed above 3.31 discards a correct match on a German ingredient name;
any floor below 10.18 admits `UNSPECIFIED INGREDIENT` resolving to caviar. There
is no cut that separates them, because the quantity being cut is not measuring
what the cut is meant to select on.

Calibrating that floor against hand-labelled data would spend annotation on a
frame already shown to be wrong. This ADR exists so the frame is chosen first.

## Candidate signals

**1. The rung.** Which resolution step produced the match: curated override,
`prod_ai` exact, `drugname` exact, `prod_ai` approximate, `drugname` approximate.
Categorical, free, already recorded per row. An exact term match is a different
kind of claim from an approximate one, and lumping them under one numeric
threshold is what makes the threshold look necessary.

**2. Rank at which ingredient resolution succeeded.** The probe found that the
top-ranked candidate can resolve to no ingredient at all: for
`HYDROCODONE BITARTRATE AND ACETAMINOPHEN`, rank 1 returned empty `IN` and `PIN`
groups and rank 2 resolved correctly. A match that needed rank 4 is weaker
evidence than one that resolved at rank 1. Free, and already produced by the
resolution walk.

**3. Token overlap between the query and the resolved ingredient name.** Cheap,
local, and it catches the failure the score cannot see: `UNSPECIFIED INGREDIENT`
and `caviar preparation` share no token, while `ACETYLSALICYLSAEURE` and
`aspirin` share none either. That second case is the warning - this signal is
strong against fabricated matches and weak against non-English synonyms, so it
discriminates in one direction only and cannot be used alone.

**4. Score normalized by query length.** If the raw score tracks length, dividing
it out may restore comparability across strings. Untested, and testable without
any annotation: correlate score against token count over a random sample of
resolved matches and see whether the residual separates the known-bad cases.

**5. Agreement between the two source fields.** 77,927,677 drug rows carry both
`prod_ai` and `drugname`. Resolving each independently and comparing the
ingredients gives corroboration on 92.12 percent of rows **with no human
labelling at all**. Disagreement is not proof of error, but the disagreement rate
per rung is a measured upper bound on the error rate of that rung, and it is
available today. Its weakness is exactly where the problem is: `prod_ai` does not
exist before 2014Q3, so this signal is silent on the 471,074 strings that never
appear beside one.

## What would discriminate between them, and what it costs

- **Signal 5 costs nothing but compute.** One pass over rows carrying both
  fields, producing a per-rung agreement rate. It should be measured before any
  annotation is requested, because it may make the ranking obvious.
- **Signal 4 costs nothing but compute.** A correlation and a residual plot over
  resolved matches.
- **Signals 1, 2 and 3 need labels to be weighed against each other**, but far
  fewer than a floor calibration would: a sample stratified by rung and by
  resolution rank, rather than by a score whose bands are not comparable across
  strings. The size follows from the number of cells, not from the score range.
- **The 471,074 strings with no `prod_ai` are the population that matters**, and
  no cheap signal covers them. Whatever is chosen has to be evaluated there
  specifically, not on the corpus average, or it will be validated on the easy
  92 percent and deployed on the hard 8.

## Decision

None. Hassan chooses the mechanism. Until then `Settings.rxnorm_score_floor`
exists as a plumbing placeholder and is not presented as a confidence measure,
and no floor sample is emitted.

## Consequences

Recorded when it is decided.
