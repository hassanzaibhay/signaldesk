"""Celery application.

Three queues, because the work has three different shapes: ``ingest`` is long and
IO-bound, ``rag`` is short and model-bound, ``default`` is everything else.
Separating them means a six-hour download cannot starve a retrieval request.

``acks_late`` is on: a task is acknowledged after it finishes, so a worker killed
mid-task leaves the message on the broker instead of losing it. Every task
therefore has to be idempotent, which is a project rule anyway.
"""

from __future__ import annotations

import os

from celery import Celery  # type: ignore[import-untyped]

from signaldesk.core.config import get_settings

TASK_TIME_LIMIT_SECONDS = 60 * 60 * 6
TASK_SOFT_TIME_LIMIT_SECONDS = TASK_TIME_LIMIT_SECONDS - 300

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "signaldesk.web.config.settings.dev")

settings = get_settings()

app = Celery("signaldesk")
app.conf.update(
    broker_url=settings.celery_broker_url,
    result_backend=settings.celery_result_backend,
    task_default_queue="default",
    task_queues={
        "default": {"exchange": "default", "routing_key": "default"},
        "ingest": {"exchange": "ingest", "routing_key": "ingest"},
        "rag": {"exchange": "rag", "routing_key": "rag"},
    },
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_track_started=True,
    task_time_limit=TASK_TIME_LIMIT_SECONDS,
    task_soft_time_limit=TASK_SOFT_TIME_LIMIT_SECONDS,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=200,
    broker_connection_retry_on_startup=True,
    timezone="UTC",
    enable_utc=True,
)
app.autodiscover_tasks(["signaldesk"])
