#!/bin/sh
set -eu
# Compose healthchecks cover the normal path; this guards against races on slow hosts.
python - <<'PY'
import os, socket, time
host = os.environ.get("POSTGRES_HOST", "postgres")
port = int(os.environ.get("POSTGRES_PORT", "5432"))
for _ in range(60):
    try:
        socket.create_connection((host, port), timeout=2).close()
        break
    except OSError:
        time.sleep(1)
else:
    raise SystemExit(f"postgres unreachable at {host}:{port}")
PY
exec "$@"
