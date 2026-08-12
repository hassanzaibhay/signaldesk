"""Health endpoint against the real stack.

Requires the compose services; run with ``make test`` inside the container.
"""

from __future__ import annotations

import json

import pytest
from django.test import Client

from signaldesk import __version__

pytestmark = [pytest.mark.integration, pytest.mark.django_db]


def test_healthz_reports_every_dependency_reachable() -> None:
    response = Client().get("/healthz/")
    body = json.loads(response.content)

    assert response.status_code == 200, body
    assert body["status"] == "ok"
    assert body["version"] == __version__
    assert body["checks"]["database"]["ok"] is True
    assert body["checks"]["redis"]["ok"] is True
    assert body["checks"]["migrations"]["ok"] is True


def test_openapi_schema_is_served() -> None:
    response = Client().get("/api/schema/")
    assert response.status_code == 200
    assert b"openapi" in response.content
