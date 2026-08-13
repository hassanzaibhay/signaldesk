# 6. Track Django 5.2 LTS, and leave the pin permissive

Date: 2026-08-13

## Status

Accepted

## Context

The dependency list pins `django~=5.1`. A two-component compatible-release
specifier means `>=5.1, ==5.*`, so it permits the whole 5.x series, and the
image resolved Django 5.2.17 rather than a 5.1 release. The stack table said
Django 5.1.

That looked at first like a pin failing to hold. It is not: the specifier is
behaving exactly as written, and the same is true of the other two-component
pins in the file, which resolved ruff 0.16 against `~=0.6` and mypy 1.19 against
`~=1.11`.

The question is therefore which version this project should be on, not whether
the pin leaked.

## Decision

Track Django 5.2, and record it as the version in the stack table.

5.2 is a long-term support release; 5.1 is not. This repository is a portfolio
artifact that will sit untouched between bursts of work, and an LTS release is
supported for three years rather than eight months. Being pinned to a
short-support release would mean that picking the project up after a gap starts
with a framework upgrade rather than with the work.

The pin stays as `django~=5.1` rather than tightening to `~=5.2.0`. Within a
major series, Django's compatibility policy is strong, the test suite runs on
every change, and a floating patch level is how security fixes arrive without a
manual bump. The version that matters for reproducibility is the one recorded in
`uv.lock`, which is committed and exact; the specifier only sets the range the
lock may be resolved within.

## Consequences

Anyone reading the stack table sees the version actually installed. Anyone
reading the specifier should understand it as a range, with `uv.lock` as the
authority on what is installed.

The general point applies to every two-component pin in this project: they are
ranges, not versions, and reporting a resolved version as a violation of one is
a misreading. Where an exact version genuinely matters, the lock file is the
mechanism, not the specifier.
