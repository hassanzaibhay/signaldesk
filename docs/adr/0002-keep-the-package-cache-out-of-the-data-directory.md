# 2. Keep the package cache out of the data directory

Date: 2026-08-12

## Status

Accepted

## Context

The image sets `XDG_CACHE_HOME=/data/cache` so that model weights and downloaded
artifacts land in the `signaldesk_data` volume, which exists because bind mounts
on Windows are too slow for directories full of large files. It also sets
`HF_HOME=/data/models` for the same reason.

The dependency install steps mount a build cache at `/root/.cache/uv`:

```
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system ".[ml]"
```

That target is wrong once `XDG_CACHE_HOME` is set. The package manager derives
its cache directory from `XDG_CACHE_HOME`, so it writes to `/data/cache/uv`, not
to `/root/.cache/uv`. Three consequences, all measured on a cold build:

* The build cache mount caches nothing, so every build re-downloads the wheels.
* 1.8 GB of wheel cache is committed into the image at `/data/cache/uv`. The
  named volume shadows that path at runtime, so the content is unreachable as
  well as unwanted: `du -sh /data` inside the built image reported 1.8 GB.
* The final stage runs `chown -R app:app /app /data`, which then has to rewrite
  ownership on every one of those cached files. That single step took 554
  seconds and produced a 1.83 GB layer.

The image measured 1.59 GB of content and 7.51 GB of disk usage, against a
target of 2.5 GB.

## Decision

Set `UV_CACHE_DIR=/root/.cache/uv` explicitly in the base stage, alongside the
existing environment block.

The cache location is now stated rather than inherited, so the build cache mount
covers it, `/data` ships empty, and the ownership fix at the end of the build has
only the application tree to walk. `XDG_CACHE_HOME` and `HF_HOME` are unchanged:
model weights and runtime caches still belong in the data volume, which was the
point of setting them.

## Consequences

The image loses the baked-in wheel cache and the layer that chowned it. Repeat
builds reuse the mounted cache instead of re-downloading, and the ownership step
completes in seconds rather than in nine minutes.

The general lesson is worth stating because it will recur: any tool whose cache
location is derived from `XDG_CACHE_HOME` needs its build-time cache mount
pointed at the derived path, or its cache directory pinned explicitly. Inherited
paths and explicit mount targets have to be reconciled deliberately.
