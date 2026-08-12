"""Celery is configured for long, resumable work on three separate queues."""

from __future__ import annotations

import pytest

from signaldesk.core.celery_app import (
    TASK_SOFT_TIME_LIMIT_SECONDS,
    TASK_TIME_LIMIT_SECONDS,
    app,
)

pytestmark = pytest.mark.unit


def test_three_queues_are_declared() -> None:
    assert set(app.conf.task_queues) == {"default", "ingest", "rag"}


def test_tasks_are_acknowledged_late() -> None:
    assert app.conf.task_acks_late is True
    assert app.conf.task_reject_on_worker_lost is True


def test_time_limits_leave_room_for_a_soft_shutdown() -> None:
    assert app.conf.task_time_limit == TASK_TIME_LIMIT_SECONDS
    assert app.conf.task_soft_time_limit == TASK_SOFT_TIME_LIMIT_SECONDS
    assert TASK_SOFT_TIME_LIMIT_SECONDS < TASK_TIME_LIMIT_SECONDS


def test_workers_take_one_task_at_a_time() -> None:
    assert app.conf.worker_prefetch_multiplier == 1
