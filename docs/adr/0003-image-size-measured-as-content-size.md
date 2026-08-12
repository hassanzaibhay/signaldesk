# 3. Measure image size as content size

Date: 2026-08-12

## Status

Accepted

## Context

The image has a size budget, and the budget was written against the single
`SIZE` column that `docker image ls` used to print. Docker 29 replaced that
column with two:

```
IMAGE              ID             DISK USAGE   CONTENT SIZE   EXTRA
signaldesk:local   f2e3e9dfe601        2.9GB          636MB
```

The two measure different things. Content size is the image itself: the sum of
its layer blobs, which is what a pull transfers and what a registry stores.
Disk usage additionally counts the unpacked snapshots the local daemon keeps so
containers can run, plus shared base layers. Unpacking roughly doubles the
figure by construction, so disk usage is a property of the local daemon's
storage, not of the artifact being measured. It also double-counts across
images that share a base.

Reporting disk usage would make the budget depend on the machine that ran the
build and on what else is in its image store, which is not a useful gate.

## Decision

The image size budget is measured as content size, and reads: content size
under 2.5 GB.

`docker image inspect --format '{{.Size}}'` reports the same number in bytes and
is the scriptable form for continuous integration.

## Consequences

The current image measures 636 MB of content against a 2.5 GB budget, with the
headroom that later work will spend on model weights and index artifacts. Disk
usage is still worth watching, but as a host capacity question rather than as a
property of the image: see the disk budget in the data integrity conventions.

Anyone comparing this number against an older report should check which column
it came from. Under the pre-29 output the same image would have printed a single
figure closer to the disk usage side, so the two are not comparable without
saying which was meant.
