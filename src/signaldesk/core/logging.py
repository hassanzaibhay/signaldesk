"""Structured logging.

One configuration for the whole project: JSON to stdout, ISO timestamps, the
logger name and level on every record. Application code binds key-value pairs
rather than interpolating strings, so logs stay queryable once they are shipped
somewhere:

    log.info("faers.quarter.loaded", quarter=quarter, cases=n, seconds=elapsed)
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

_configured = False


def configure_logging(*, debug: bool = False) -> None:
    """Configure structlog and the standard library root logger.

    Idempotent: calling it twice does not stack processors. ``debug`` only
    lowers the level; the output stays JSON in every environment so that
    development and production logs are the same shape.
    """
    global _configured

    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level, force=True)

    if _configured:
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(level),
            cache_logger_on_first_use=True,
        )
        return

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        # No explicit file: PrintLogger then resolves sys.stdout at write time,
        # which keeps working when something replaces the stream underneath it.
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound logger. Prefer module-level ``log = get_logger(__name__)``."""
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger


def django_logging_config(*, debug: bool = False) -> dict[str, Any]:
    """The ``LOGGING`` dict for Django settings.

    Django's own loggers are routed through the same JSON handler so a request
    log line and an ingest log line are parseable by the same consumer.
    """
    level = "DEBUG" if debug else "INFO"
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": structlog.stdlib.ProcessorFormatter,
                "processor": structlog.processors.JSONRenderer(),
                "foreign_pre_chain": [
                    structlog.processors.add_log_level,
                    structlog.processors.TimeStamper(fmt="iso", utc=True),
                ],
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "json",
                "stream": sys.stdout,
            }
        },
        "root": {"handlers": ["console"], "level": level},
        "loggers": {
            "django": {"handlers": ["console"], "level": level, "propagate": False},
            "django.db.backends": {
                "handlers": ["console"],
                "level": "WARNING",
                "propagate": False,
            },
            "signaldesk": {"handlers": ["console"], "level": level, "propagate": False},
        },
    }
