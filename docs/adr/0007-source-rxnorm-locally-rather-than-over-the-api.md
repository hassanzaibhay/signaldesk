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

Requests per string is measured rather than assumed. Fifteen strings drawn from
the corpus by row count, run through the real matcher against the real endpoint,
cost **2.07 requests each**: an exact-name lookup plus one ingredient resolution
for most, four for a two-component combination, and zero for
`HUMIRA 40 MG/0.8 ML PEN`, which cleans to a query already cached. Fourteen of the
fifteen were answered by the exact rung.

At 2.07 requests per string, 538,917 strings is about **1.12 million requests, or
15.5 hours** at the published ceiling. Measured latency is 0.53 s warm, so a
single-threaded pass is nearer 160 hours; reaching the ceiling needs about ten
concurrent connections held for the whole run. An earlier draft quoted 7.5 hours,
which counted the approximate calls alone and therefore about half the work.
Neither figure is inside the four-hour gate this project sets for a pipeline
stage, and concurrency cannot be raised past the published limit to close the gap.

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

| | Approximate matching | Rate limit | Local disk | Time to first figure |
|---|---|---|---|---|
| A. RxNav-in-a-Box | yes | none | 100 GB, one-time | ~1 h import, then fast |
| B. Release files | no | none | small | fast, but semantics change |
| C. Stay on the API | yes | 20/s | 1.1 GB of cache | 15.5 h, once |

A and C differ in resources and elapsed time, not in what they can compute. B is
the only one that changes the answer, and it is therefore the one that needs the
strongest justification rather than the weakest.

## The reproducibility argument does not survive its own measurement

An earlier draft weighed a property in option C's favour: a cold clone reproduces
the pipeline from the committed cache with no account anywhere. That property was
asserted at fixture scale and never checked at corpus scale, so it was measured.

The same fifteen-string run wrote 31 cache entries totalling 31,568 bytes, a mean
of 1,018 bytes per entry and **2,105 bytes per string**. Projected across the
corpus:

| | Entries | On disk |
|---|---|---|
| Minimum, 538,917 strings | ~1.12 M | **1.06 GiB** |
| Upper, 656,589 strings | ~1.36 M | **1.29 GiB** |

**That is not committable.** Not because of raw size alone - a `tar | gzip` of the
sample compresses about six to one, so the archive might land in the low hundreds
of megabytes - but because it is over a million small files in a repository whose
entire tracked content is currently under a megabyte, and because the rule against
committing downloaded data covers exactly this.

So the reproducibility property has **already lapsed at full scale, under every
option**. A cold clone cannot reproduce a full-corpus normalization from anything
committed here today. What that changes:

- Option C's advantage was never "reproducible for anyone", it was "no credential
  needed for the person who runs it", and that person pays 15.5 hours once.
- Option A's 100 GB stops being a standing requirement on anyone reproducing the
  work and becomes a **one-time local cost to populate a cache**, paid by the same
  single person, on the same single machine. It buys back the 15.5 hours and the
  dependency on NLM's uptime.

The comparison is therefore between two one-time local costs, not between a
portable option and an unportable one.

**Distribution plan for the cache, whichever option wins.** The fixture-scale
slice stays committed so CI keeps running with no network and no API keys, exactly
as it does now. The full-corpus cache is local state, listed alongside the other
uncommitted artifacts in `docs/data-sources.md`, and the coverage report records
the entry count and byte size so its absence is visible rather than silent. No
plan here involves committing it or hosting it.

## What remains to be established

- **Whether the RxNorm service can be deployed alone.** The 100 GB and 12 GB
  figures cover the whole composition: RxNav, RxClass and RxMix as applications
  plus four APIs. Only the RxNorm API is needed here. NLM publishes no per-service
  breakdown, and the README offers only "you may take the included
  docker-compose.yml file as an example", which implies the services are separable
  without saying what a subset costs. Establishing it requires downloading the
  distribution, which requires the licence agreement, so it is a precondition of
  adopting A rather than an input available now. **If a RxNorm-only subset is
  materially smaller, the disk objection to A largely dissolves.**
- **Licensing.** A and B both need a UMLS account and license agreement, free of
  charge, so the zero-cost constraint holds either way. A's "individual, personal
  use" framing should be read against this being a public portfolio repository:
  the artifact is not redistributed, but the point is worth a deliberate reading
  rather than an assumption.
- **CI.** Runs with no API keys under every option, so the fixture-scale path must
  keep working against a mock whichever one wins.

## Decision

None yet. Hassan rules.

## Consequences

Recorded when it is decided.
