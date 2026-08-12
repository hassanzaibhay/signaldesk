"""Logging emits JSON with bound key-value pairs, and configuring twice is safe."""

from __future__ import annotations

import json

import pytest

from signaldesk.core.logging import configure_logging, django_logging_config, get_logger

pytestmark = pytest.mark.unit


def test_events_render_as_json_with_bound_pairs(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging()
    get_logger("test").info("faers.quarter.loaded", quarter="2013Q1", cases=42)

    record = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert record["event"] == "faers.quarter.loaded"
    assert record["quarter"] == "2013Q1"
    assert record["cases"] == 42
    assert record["level"] == "info"
    assert "timestamp" in record


def test_configuring_twice_does_not_duplicate_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging()
    configure_logging(debug=True)
    get_logger("test").info("once")

    lines = [line for line in capsys.readouterr().out.strip().splitlines() if line]
    assert len(lines) == 1


def test_debug_events_are_dropped_at_the_default_level(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging(debug=False)
    get_logger("test").debug("not.emitted")
    assert capsys.readouterr().out.strip() == ""


def test_django_logging_config_routes_project_loggers() -> None:
    config = django_logging_config(debug=True)
    assert config["loggers"]["signaldesk"]["level"] == "DEBUG"
    assert config["handlers"]["console"]["formatter"] == "json"
    assert config["loggers"]["django.db.backends"]["level"] == "WARNING"
