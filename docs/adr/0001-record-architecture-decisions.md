# 1. Record architecture decisions

Date: 2026-08-12

## Status

Accepted

## Context

This project makes a number of decisions that are cheap to make once and
expensive to revisit: the unit of statistical analysis, which reference standards
count as ground truth, whether retrieval is hybrid, which datastore holds the
vectors, and what the project refuses to claim. Six months later the reasoning
behind such a decision is usually gone, and the decision gets re-litigated from
scratch or quietly worked around.

## Decision

Every architecturally significant decision is recorded here as a numbered
Markdown file, in the style described by Michael Nygard. A record states the
context that forced the decision, the decision itself, and its consequences.

Records are immutable once accepted. A decision that turns out to be wrong is not
edited: a new record supersedes it and says so, and the superseded record is
marked accordingly. The history of what was believed and when is part of the
value.

A change that contradicts an accepted record is not made without a record that
supersedes it.

## Consequences

Reviewing the design means reading a short, ordered list rather than
archaeology through commit messages. Changing a locked decision costs one file,
which is the right amount of friction: enough to force the argument to be
written down, not so much that a genuinely wrong decision survives.
