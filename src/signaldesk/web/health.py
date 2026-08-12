"""Liveness and readiness endpoint.

``/healthz/`` answers one question: can this process serve traffic right now.
It checks the two dependencies that make the answer no - Postgres and Redis -
and reports whether migrations are outstanding, which is the usual cause of a
process that is up but wrong. Unauthenticated, because a health check that
needs credentials is not usable by the orchestrator.
"""

from __future__ import annotations

from typing import Any

import redis
from django.db import DatabaseError, connections
from django.db.migrations.executor import MigrationExecutor
from django.http import HttpRequest, JsonResponse
from django.views.decorators.http import require_GET

from signaldesk import __version__
from signaldesk.core.config import get_settings


def _check_database() -> dict[str, Any]:
    try:
        with connections["default"].cursor() as cursor:
            cursor.execute("SELECT 1")
            cursor.fetchone()
    except (DatabaseError, OSError) as exc:
        return {"ok": False, "error": type(exc).__name__}
    return {"ok": True}


def _check_redis() -> dict[str, Any]:
    settings = get_settings()
    try:
        client: redis.Redis = redis.Redis.from_url(
            settings.redis_url, socket_connect_timeout=2, socket_timeout=2
        )
        client.ping()
        client.close()
    except (redis.RedisError, OSError) as exc:
        return {"ok": False, "error": type(exc).__name__}
    return {"ok": True}


def _check_migrations() -> dict[str, Any]:
    try:
        executor = MigrationExecutor(connections["default"])
        targets = executor.loader.graph.leaf_nodes()
        plan = executor.migration_plan(targets)
    except (DatabaseError, OSError) as exc:
        return {"ok": False, "error": type(exc).__name__}
    return {"ok": not plan, "pending": len(plan)}


@require_GET
def healthz(request: HttpRequest) -> JsonResponse:
    """Report version and dependency state. 200 when everything is reachable."""
    checks = {
        "database": _check_database(),
        "redis": _check_redis(),
        "migrations": _check_migrations(),
    }
    healthy = all(check["ok"] for check in checks.values())
    payload: dict[str, Any] = {
        "status": "ok" if healthy else "degraded",
        "version": __version__,
        "checks": checks,
    }
    return JsonResponse(payload, status=200 if healthy else 503)
