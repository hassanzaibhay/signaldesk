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

from signaldesk.core.redaction import CredentialRedactingFilter, redact_processor

_configured = False

#: One instance, added to every root handler. A filter object may be attached to
#: several handlers, and ``addFilter`` is idempotent per handler.
_REDACT = CredentialRedactingFilter()

#: Loggers pinned below INFO whatever the root level is.
#:
#: ``httpx`` logs every request at INFO with the full URL, query string included,
#: so an ingest with a key in the query publishes it once per page. The pin is on
#: the logger rather than on the root level because ``--verbose`` sets the root to
#: DEBUG and would otherwise re-expose the line it was meant to suppress.
#:
#: The pin is a level, not a redaction. It is the redaction filter that makes the
#: credential safe; this only stops several hundred uninteresting lines per run.
_QUIET_LOGGERS: dict[str, int] = {"httpx": logging.WARNING}


def configure_logging(*, debug: bool = False) -> None:
    """Configure structlog and the standard library root logger.

    Idempotent: calling it twice does not stack processors. ``debug`` only
    lowers the level; the output stays JSON in every environment so that
    development and production logs are the same shape.
    """
    global _configured

    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level, force=True)

    # Before the _configured guard, and unconditionally. basicConfig(force=True)
    # closes the previous root handler and builds a new one on every call, so a
    # filter attached on the first call is gone by the second. The level pins are
    # re-applied for the same reason: cheap, and being wrong here is silent.
    for name, pinned in _QUIET_LOGGERS.items():
        logging.getLogger(name).setLevel(pinned)
    for handler in logging.getLogger().handlers:
        handler.addFilter(_REDACT)

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
            # Last before rendering, so it sees the merged event dict rather than
            # whatever the caller bound. structlog events never become LogRecords
            # under PrintLoggerFactory, so the handler filter cannot reach them
            # and this is the only thing that does.
            redact_processor,
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

    The redaction filter is attached to the handler rather than to any logger, for
    the reason given in ``core.redaction``: a handler sees records from loggers
    that did not exist when this was written.
    """
    level = "DEBUG" if debug else "INFO"
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "redact_credentials": {"()": CredentialRedactingFilter},
        },
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
                "filters": ["redact_credentials"],
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
            # Pinned to WARNING independently of ``level``, so DEBUG here does
            # not restore the per-request line carrying the query string. Same
            # reason as _QUIET_LOGGERS, which covers the command line path.
            "httpx": {"handlers": ["console"], "level": "WARNING", "propagate": False},
            "signaldesk": {"handlers": ["console"], "level": level, "propagate": False},
        },
    }
