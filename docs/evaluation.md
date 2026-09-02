# Evaluation

Evaluation protocols, gold set construction, the annotation guideline, and judge
validation. Filled in with the evaluation harness.

## Annotation guidelines

- [Labeledness](annotation-guideline-labeledness.md) -- the verdict set, what
  counts as an event being described in a label, the reading protocol, and what
  goes in the unclear bucket. Approved before annotation began; every annotation
  record carries the guideline version it was made under.

## Gold sets

`evals/golden/labeledness_v1.jsonl` is hand-curated. No model participates in
drawing, ordering, pre-filling or suggesting any part of it. The sample it is
drawn against is committed under `evals/history/` before annotation starts, so
the frame cannot be adjusted after seeing what came out of the draw.

The harness is `signaldesk evals annotate`, which reads the committed manifest
and nothing else -- no database, no network, no model client. That is enforced by
a test that walks the import graph statically rather than by convention.
