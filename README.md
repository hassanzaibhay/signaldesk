# SignalDesk

Post-market drug safety signal intelligence.

SignalDesk ingests the FDA adverse event reporting extracts, computes
pharmacovigilance disproportionality statistics over the full corpus, and uses a
retrieval-augmented model layer to adjudicate each candidate signal against
primary regulatory documents: FDA drug labels, ClinicalTrials.gov results
sections, and PubMed literature. Every generated statement carries a citation
that resolves to an exact span in a source document.

Every layer reports its own measured accuracy against published reference
standards.

## Disproportionality is hypothesis-generating

Disproportionality statistics quantify reporting patterns in a voluntary adverse
event database. They do not establish that a drug caused an event. Reporting is
affected by publicity, litigation, time on the market, and indication, none of
which are causal mechanisms. Every screen produced here is a hypothesis for a
reviewer to assess, not a finding.

Any generated text that asserts causation is treated as a defect and fails the
evaluation suite.

## Status

Under construction. The foundation - container stack, configuration, health
checks, command line interface, continuous integration - is in place. Ingest,
statistics, retrieval, and the web surfaces arrive in later work.

## Quickstart

Requires Docker and GNU Make. Nothing else is installed on the host.

```
cp .env.example .env
make bootstrap
```

That builds the image, starts Postgres and Redis, applies migrations, starts the
application, and loads the committed fixture slice. The application is then at
http://localhost:8000/ and its health endpoint at
http://localhost:8000/healthz/.

On Windows, GNU Make comes from `winget install ezwinports.make`.

Useful targets:

```
make up            # start the stack
make down          # stop it
make test          # full test suite with coverage
make test-fast     # unit tests only
make fmt lint type # format, lint, type-check
make hygiene       # encoding and commit-metadata checks
make eval-all      # every evaluation suite
```

## Results

Pending. Every number this project publishes is produced by a committed
evaluation run under `evals/history/` and is reproducible from this repository.
No estimated figures appear here: a claim that cannot be traced to an artifact
does not go in the README.

The metrics that will appear here are the disproportionality validation against
published reference standards, retrieval quality with the ablation table behind
it, labeledness classification against a hand-annotated gold set, citation
precision, judge agreement with a human annotator, and latency and cost per
query.

## Data sources and licensing

Source provenance, licence terms, and refresh cadence are documented in
`docs/data-sources.md`. The repository ships no MedDRA content: adverse event
Preferred Terms arrive with the public FDA extracts, and the optional hierarchy
loader expects a user-supplied release.

## Licence

Apache-2.0. See `LICENSE`.
