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

**A. RxNav-in-a-Box.** A Docker composition of the RxNav, RxClass and RxMix
applications together with the RESTful APIs, RxNorm among them. The client code
does not change beyond a base URL, and the rate limit disappears with the network.

**It exposes the approximate matcher.** This was checked rather than assumed,
because the answer decides whether this option loses a capability. NLM's public
README for the distribution names the function explicitly:

> The API functions getApproximateMatch and getProprietaryInformation in
> RxNav-in-a-Box do not accept a UTS API key.

and the RxNorm API index maps that function to the endpoint this project uses:

> getApproximateMatch  /approximateTerm  Concept and atom IDs approximately
> matching a query

So option A is capability-complete. An earlier draft of this ADR claimed the
choice was between the API and exact-matching-only. That framing was wrong, it
came from conflating this option with option B, and the conclusion it pointed
towards has been withdrawn.

What it costs is published in the same README, and it is the real constraint:

> - 12 gigabytes of memory to devote to a container platform (e.g., Docker)
> - 100 gigabytes of disk space

with `docker compose up` taking "about an hour to import the included RxNav
data". The distribution is "designed for individual, personal use", and the
download requires a UMLS license agreement.

**Measured against this machine, that does not currently fit.** After reclaiming
the Docker virtual disk the host has 70.2 GB free against a 100 GB requirement,
and the Docker VM is allocated 11.69 GB of memory against a 12 GB requirement for
this composition alone, before SignalDesk's own Postgres, Redis and workers. Both
are movable numbers - disk can be freed, the VM limit can be raised on a machine
with the RAM - but neither is free, and the disk figure is larger than the entire
projected footprint of the rest of the project.

**B. RxNorm full release files.** The RRF distribution, UTF-8, with technical
documentation and load scripts, and a documented path for automating downloads.
Loading `RXNCONSO` and `RXNREL` into the analytical store gives exact term
matching and ingredient resolution directly in SQL, with no service to run and no
100 GB image.

This is the option that loses approximate matching. `getApproximateMatch` is an
NLM application function, not a column in the release, so choosing B means either
accepting exact matching only - which the corpus will not tolerate, given 471,074
strings that never appear beside a `prod_ai` - or writing a substitute scorer,
which changes the match semantics and puts this project in the business of string
similarity rather than pharmacovigilance.

**C. Stay on the API.** Honest, and possible if the pass is treated as a
multi-day background job resumable from the on-disk cache. The cache makes the
second run free, so the cost is paid once, and the rung order already keeps most
strings away from the expensive endpoint: exact matching answers first and costs
the same one request. It also means P03 cannot report a coverage figure for days.

## The shape of the choice

Three options, not two, and the trade is no longer capability against convenience:

| | Approximate matching | Rate limit | Disk | Time to first figure |
|---|---|---|---|---|
| A. RxNav-in-a-Box | yes | none | 100 GB | ~1 h import, then fast |
| B. Release files | no | none | small | fast, but semantics change |
| C. Stay on the API | yes | 20/s | none | days, once |

A and C differ in resources and elapsed time, not in what they can compute. B is
the only one that changes the answer, and it is therefore the one that needs the
strongest justification rather than the weakest.

## What remains to be established

- **Licensing.** A and B both need a UMLS account and license agreement, free of
  charge, so the zero-cost constraint holds either way. A's "individual, personal
  use" framing should be read against this being a public portfolio repository:
  the artifact is not redistributed, but the point is worth a deliberate reading
  rather than an assumption.
- **Reproducibility.** Today a cold clone reproduces the pipeline from the
  committed cache and fixtures with no account anywhere. A and B both introduce a
  credentialed download, and neither artifact can be committed. CI runs with no
  API keys, so the fixture-scale path must keep working against a mock whichever
  option wins.

## Decision

None yet. Hassan rules.

## Consequences

Recorded when it is decided.
