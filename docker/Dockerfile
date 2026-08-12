# syntax=docker/dockerfile:1.7
FROM python:3.12-slim-bookworm AS base
# UV_CACHE_DIR is stated rather than inherited from XDG_CACHE_HOME, so the build
# cache mounts below cover it and no wheel cache is committed under /data.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    HF_HOME=/data/models \
    XDG_CACHE_HOME=/data/cache \
    UV_CACHE_DIR=/root/.cache/uv
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential curl git libpq5 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.4 /uv /usr/local/bin/uv
WORKDIR /app

FROM base AS deps
COPY pyproject.toml uv.lock* README.md ./
# The package is installed from source; hatchling needs the tree to exist at this
# stage. The real source arrives with COPY . /app in the runtime stage.
RUN mkdir -p src/signaldesk && touch src/signaldesk/__init__.py
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system torch --index-url https://download.pytorch.org/whl/cpu
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system ".[ml]"
# The dev extra mirrors the dev dependency group: "uv pip install --group" does
# not exist in the uv releases pinned above, and extras do work there.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system ".[ml,dev]"

FROM deps AS runtime
COPY . /app
RUN uv pip install --system --no-deps -e . \
    && groupadd -r app && useradd -r -g app -d /app app \
    && mkdir -p /data && chown -R app:app /app /data
USER app
EXPOSE 8000
ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
