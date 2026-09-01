"""Logging emits JSON with bound key-value pairs, and configuring twice is safe.

The redaction tests are the substantial half. A credential reaches a log because
some library logs a URL, and the library that does it today is httpx; the library
that does it next has not been written yet. So what is asserted is not that httpx
is quiet, but that a record carrying a credential comes out redacted whichever
logger produced it, and that the same holds once Django has applied its own
configuration over the top.
"""

from __future__ import annotations

import json
import logging
import logging.config
from collections.abc import Iterator

import httpx
import pytest

from signaldesk.core.logging import configure_logging, django_logging_config, get_logger
from signaldesk.core.redaction import CredentialRedactingFilter

pytestmark = pytest.mark.unit

#: Distinctive enough that a substring check cannot pass by accident.
FAKE_KEY = "sk-test-DEADBEEF0123456789abcdef"

#: The line httpx actually emits, format string and argument types included.
#: request.url is an httpx.URL, not a str, which is the detail an isinstance
#: guard in the filter would miss.
HTTPX_FORMAT = 'HTTP Request: %s %s "%s %d %s"'


@pytest.fixture
def restore_logging() -> Iterator[None]:
    """Put the process's logging back afterwards.

    configure_logging pins levels and attaches a handler filter globally, so
    without this a redaction test would silence httpx for every test that runs
    after it. A test must not change what its neighbours measure.
    """
    root = logging.getLogger()
    handlers = list(root.handlers)
    level = root.level
    pinned = logging.getLogger("httpx").level
    try:
        yield
    finally:
        root.handlers = handlers
        root.setLevel(level)
        logging.getLogger("httpx").setLevel(pinned)


def _probe_url() -> httpx.URL:
    return httpx.URL(f"https://api.fda.gov/drug/label.json?search=x&limit=100&api_key={FAKE_KEY}")


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


class TestCredentialRedaction:
    """The api_key must not reach stdout, whatever emits it."""

    def test_httpx_is_pinned_below_info_even_under_verbose(self, restore_logging: None) -> None:
        """--verbose sets the root to DEBUG and must not re-expose the line.

        httpx logs every request at INFO with the full URL. The pin is on the
        logger rather than on the root level for exactly this case: a level check
        against the root would pass here and the line would still be emitted.
        """
        configure_logging(debug=True)
        assert logging.getLogger().isEnabledFor(logging.DEBUG)
        assert not logging.getLogger("httpx").isEnabledFor(logging.INFO)

    def test_a_credential_is_redacted_from_a_logger_nobody_pinned(
        self, restore_logging: None, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The point of filtering at the handler rather than at a logger.

        This record is shaped exactly like httpx's, down to the URL being an
        httpx.URL object rather than a string, but it comes from a logger no
        pin covers - which is the situation the next library to log a URL will
        create.
        """
        configure_logging(debug=True)
        logging.getLogger("some.library.nobody.pinned").info(
            HTTPX_FORMAT, "GET", _probe_url(), "HTTP/1.1", 200, "OK"
        )

        out = capsys.readouterr().out
        assert FAKE_KEY not in out
        assert "api_key=REDACTED" in out
        # The rest of the line survives: redaction, not suppression.
        assert "search=x" in out

    def test_a_credential_interpolated_by_the_caller_is_redacted_too(
        self, restore_logging: None, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """record.msg carries it when the caller formatted the string itself.

        Rewriting only the args would leave this one, and rewriting only msg
        would leave the httpx case. Both halves or neither.
        """
        configure_logging(debug=True)
        logging.getLogger("some.library").warning(f"giving up on ?api_key={FAKE_KEY}&retry=3")

        out = capsys.readouterr().out
        assert FAKE_KEY not in out
        assert "retry=3" in out

    def test_every_root_handler_carries_the_filter(self, restore_logging: None) -> None:
        """basicConfig(force=True) builds a new handler on every call.

        A filter attached once, on the first call, is gone by the second. This
        catches that and catches a handler added inside configure_logging without
        one. It does not catch a handler added at runtime afterwards, which
        remains the hole in this mechanism.
        """
        configure_logging()
        configure_logging(debug=True)
        handlers = logging.getLogger().handlers
        assert handlers
        for handler in handlers:
            assert any(isinstance(item, CredentialRedactingFilter) for item in handler.filters)

    def test_a_structlog_event_is_redacted_although_no_handler_sees_it(
        self, restore_logging: None, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """PrintLoggerFactory never produces a LogRecord.

        So the handler filter cannot reach a structlog event, and without the
        processor the filter would be claiming coverage it does not have. Both
        a bound credential key and one interpolated into a value.
        """
        configure_logging()
        get_logger("test").info("spl.probe", api_key=FAKE_KEY, url=f"?api_key={FAKE_KEY}")

        out = capsys.readouterr().out
        assert FAKE_KEY not in out
        record = json.loads(out.strip().splitlines()[-1])
        assert record["api_key"] == "REDACTED"
        assert record["url"] == "?api_key=REDACTED"

    def test_the_process_is_still_safe_after_django_applies_its_configuration(
        self, restore_logging: None, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Asserted against the resulting process state, not against the dict.

        A dict assertion says the configuration names a filter. It does not say
        that applying it produces a process where the credential is redacted and
        httpx is quiet, which is the property that matters and the one Django's
        dictConfig could undo.
        """
        logging.config.dictConfig(django_logging_config(debug=True))

        assert not logging.getLogger("httpx").isEnabledFor(logging.INFO)

        logging.getLogger("some.library").info(
            HTTPX_FORMAT, "GET", _probe_url(), "HTTP/1.1", 200, "OK"
        )
        out = capsys.readouterr().out
        assert FAKE_KEY not in out
        assert "api_key=REDACTED" in out

    def test_django_is_still_wired_to_this_module(self) -> None:
        """The one line that makes any of the above true for the web process.

        Without this, deleting ``LOGGING = django_logging_config(...)`` from the
        settings module leaves every other test in this class green while the
        web process logs credentials in full.
        """
        from signaldesk.core.config import get_settings
        from signaldesk.web.config.settings import base

        expected = django_logging_config(debug=get_settings().django_debug)
        # Not whole-dict equality: the formatters hold freshly built structlog
        # processor objects, which never compare equal across two calls. The
        # parts that carry the wiring are plain data and do compare.
        assert base.LOGGING["loggers"] == expected["loggers"]
        assert base.LOGGING["root"] == expected["root"]
        assert base.LOGGING["filters"] == expected["filters"]
        assert base.LOGGING["handlers"]["console"]["filters"] == ["redact_credentials"]
        assert base.LOGGING["filters"]["redact_credentials"]["()"] is CredentialRedactingFilter
