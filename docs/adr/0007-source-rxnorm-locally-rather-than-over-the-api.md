# 7. Source RxNorm locally rather than over the API

Date: 2026-08-14

## Status

Proposed. Drafted for a decision, not adopted.

## Context

P03 maps every FAERS drug string to an RxNorm ingredient. The work is measured:

| | |
|---|---|
| Drug rows | 84,589,952 |
| Distinct `drugname`, folded | 641,070 |
| Distinct `prod_ai`, folded | 15,519 |
| Rows with no `prod_ai` | 6,662,275, carrying 523,398 distinct strings |
| Strings never seen beside a `prod_ai` | 471,074 |

So at minimum 538,917 distinct strings need a lookup, and up to 656,589 if every
`drugname` needs its own. Each one is at least one `approximateTerm` call, plus a
`related` call per distinct concept to reach the ingredient, plus extra calls
where the top-ranked candidate resolves to no ingredient at all, which the probe
found happens.

NLM's terms of service, quoted from the page:

> In order to avoid overloading the RxNav servers, NLM requires that users of the
> APIs (RxNorm, RxTerms, Prescribable RxNorm, and RxClass) send **no more than 20
> requests per second** per IP address.

At exactly that ceiling, 538,917 approximate calls take 7.5 hours before any
ingredient resolution. Measured latency from the probe is 0.53 s warm for
`approximateTerm`, so a single-threaded pass is nearer 80 hours; reaching the
ceiling needs about ten concurrent connections held for the whole run. Neither
figure is inside the four-hour gate this project sets for a pipeline stage, and
concurrency cannot be raised past the published limit to close the gap.

The same terms page names the answer:

> If you have a specific use case that requires you to send a large number of
> requests to one of our APIs, and thus exceed the request rate limit outlined in
> this policy, please consider using RxNav-in-a-Box, the locally installable
> version of our RESTful APIs based on Docker containers.

This is not a workaround. It is the sanctioned path for this exact case.

## Options

**A. RxNav-in-a-Box.** Docker containers publishing the same REST API locally.
The client code does not change beyond a base URL, `approximateTerm` keeps its
behaviour, and the rate limit disappears with the network. Download requires a
UMLS license agreement, and the system requirements live in a `README.txt` inside
the zip rather than on the page, so they have to be read before adoption. Latest
edition at the time of writing is `rxnav-in-a-box-20260706.zip`.

**B. RxNorm full release files.** The RRF distribution, UTF-8, with technical
documentation and load scripts for Oracle and MySQL, and a documented path for
automating downloads. Loading `RXNCONSO` and `RXNREL` into the analytical store
gives exact term matching and ingredient resolution directly in SQL, with no
service to run.

The distinction that matters: **option B does not include approximate matching.**
`approximateTerm` is NLM's own algorithm, not a column in the release files.
Choosing B means either accepting exact matching only, which the corpus will not
tolerate given 471,074 strings that never appear beside a `prod_ai`, or writing a
substitute scorer, which changes the match semantics and puts this project in the
business of string similarity rather than pharmacovigilance.

**C. Stay on the API.** Honest, and possible if the pass is treated as a
multi-day background job resumable from the on-disk cache. The cache makes the
second run free, so the cost is paid once. It also means P03 cannot report a
coverage figure for weeks.

## What has to be established before this is decided

- **Disk.** Neither the image size nor the RAM requirement is published on the
  page. The rules make disk a first-class constraint and the budget table has no
  RxNorm line, so this is measured, not assumed.
- **Licensing.** Both A and B need a UMLS account and license agreement. Both are
  free of charge, so the zero-cost constraint holds either way.
- **Reproducibility.** Today a cold clone reproduces the pipeline from the
  committed cache and fixtures with no account anywhere. A and B both introduce a
  credentialed download, and neither artifact can be committed. CI runs with no
  API keys, so the fixture-scale path has to keep working against a mock
  regardless of which option wins.

## Decision

None yet. Hassan rules.

## Consequences

Recorded when it is decided.
