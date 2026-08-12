"""Health endpoint: honest status codes, and no dependency of its own.

The checks are stubbed here so the test needs neither a database nor a Redis;
the wiring against real services is covered by the integration test.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
import redis
from django.db import DatabaseError
from django.test import RequestFactory
from django.urls import reverse

from signaldesk import __version__
from signaldesk.web import health

pytestmark = pytest.mark.unit


def _response_body(response: Any) -> dict[str, Any]:
    body: dict[str, Any] = json.loads(response.content)
    return body


def test_public_routes_are_wired() -> None:
    assert reverse("healthz") == "/healthz/"
    assert reverse("schema") == "/api/schema/"
    assert reverse("swagger-ui") == "/api/schema/swagger-ui/"


def test_all_checks_green_returns_200(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("_check_database", "_check_redis", "_check_migrations"):
        monkeypatch.setattr(health, name, lambda: {"ok": True})

    response = health.healthz(RequestFactory().get("/healthz/"))
    body = _response_body(response)

    assert response.status_code == 200
    assert body["status"] == "ok"
    assert body["version"] == __version__
    assert set(body["checks"]) == {"database", "redis", "migrations"}


def test_an_unreachable_dependency_returns_503(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(health, "_check_database", lambda: {"ok": True})
    monkeypatch.setattr(health, "_check_redis", lambda: {"ok": False, "error": "ConnectionError"})
    monkeypatch.setattr(health, "_check_migrations", lambda: {"ok": True, "pending": 0})

    response = health.healthz(RequestFactory().get("/healthz/"))

    assert response.status_code == 503
    assert _response_body(response)["status"] == "degraded"


def test_database_check_reports_the_failure_type(monkeypatch: pytest.MonkeyPatch) -> None:
    class Failing:
        def __getitem__(self, alias: str) -> Any:
            raise DatabaseError("connection refused")

    monkeypatch.setattr(health, "connections", Failing())
    assert health._check_database() == {"ok": False, "error": "DatabaseError"}


def test_redis_check_reports_the_failure_type(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> Any:
        raise redis.ConnectionError("refused")

    monkeypatch.setattr(redis.Redis, "from_url", staticmethod(refuse))
    assert health._check_redis() == {"ok": False, "error": "ConnectionError"}


def test_redis_check_pings(monkeypatch: pytest.MonkeyPatch) -> None:
    pings: list[bool] = []

    class Stub:
        def ping(self) -> bool:
            pings.append(True)
            return True

        def close(self) -> None:
            return None

    monkeypatch.setattr(redis.Redis, "from_url", staticmethod(lambda *a, **k: Stub()))
    assert health._check_redis() == {"ok": True}
    assert pings == [True]


def test_migration_check_reports_outstanding_migrations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Graph:
        @staticmethod
        def leaf_nodes() -> list[tuple[str, str]]:
            return [("signals", "0001_initial")]

    class Loader:
        graph = Graph()

    class Executor:
        loader = Loader()

        def __init__(self, connection: object) -> None:
            self.connection = connection

        def migration_plan(self, targets: object) -> list[str]:
            return ["signals.0001_initial"]

    monkeypatch.setattr(health, "MigrationExecutor", Executor)
    assert health._check_migrations() == {"ok": False, "pending": 1}
