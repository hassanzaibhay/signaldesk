"""One list of credential parameter names, and the two things that must honour it.

A credential reaches two places it does not belong, and both are served from the
same list so they cannot drift apart:

* **Logs.** ``httpx`` logs every request at INFO as
  ``'HTTP Request: %s %s "%s %d %s"'`` with ``request.url`` as an argument, and a
  URL carries its query string. Pinning the ``httpx`` logger stops that one line;
  it does not stop the next library to do the same thing. So the redaction is
  installed as a filter on the root *handler*, where it sees every record from
  every logger, including ones added after this was written.
* **The on-disk HTTP cache key.** ``api_key`` does not change what openFDA
  returns, only how fast it may be asked, so it is not part of resource identity.
  A key in the cache key means a re-run with a different key, or with none, misses
  every entry it already holds. That is not hypothetical: the 10:16 run wrote 317
  entries keyless, the key was supplied, and every one of them became
  unaddressable.

The list is a **denylist of credential names, not a heuristic**. An unrecognised
parameter is left alone in both places, because a parameter that genuinely varies
the response must keep varying the cache key, and a value that is not a secret
should stay readable in a log. Bare ``key`` is deliberately absent: it is as
plausibly a resource identifier as a credential, and stripping it would silently
collapse distinct resources onto one cache entry.

Matching is on the parameter *name*, exact and case-insensitive. Substring
matching would take ``page_token`` for ``token``, and a pagination cursor is
exactly the kind of parameter that must keep varying the key.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from typing import Final

from structlog.typing import EventDict, WrappedLogger

#: Parameter names whose values are credentials. Exact, case-insensitive.
#:
#: Adding a name here redacts it in logs and removes it from the cache key in one
#: edit. That is the point of the shared list, and it is only a partial mechanism:
#: a credential sent under a name nobody added is still logged and still keys the
#: cache. The narrowing is that ``ingest.spl.client`` names its credential
#: parameter from this module rather than from a literal, so the two cannot
#: disagree at that one site.
CREDENTIAL_NAMES: Final[frozenset[str]] = frozenset(
    {
        "api_key",
        "apikey",
        "api-key",
        "x-api-key",
        "access_token",
        "auth",
        "auth_token",
        "client_secret",
        "password",
        "secret",
        "signature",
        "token",
    }
)

#: What replaces the value. A fixed marker rather than a mask of the right length,
#: which would leak the length.
REDACTED: Final[str] = "REDACTED"

#: ``name=value`` as it appears in a query string or an already-formatted message.
#: Built from ``CREDENTIAL_NAMES`` so there is no second list to maintain. Longest
#: name first, so ``x-api-key`` is matched whole rather than as ``api-key``.
_ASSIGNMENT: Final[re.Pattern[str]] = re.compile(
    r"\b("
    + "|".join(re.escape(name) for name in sorted(CREDENTIAL_NAMES, key=lambda n: (-len(n), n)))
    + r")=([^&\s\"'<>]*)",
    re.IGNORECASE,
)


def redact_text(text: str) -> str:
    """Rewrite every ``credential=value`` in ``text`` to ``credential=REDACTED``."""
    return _ASSIGNMENT.sub(r"\g<1>=" + REDACTED, text)


def strip_credentials(params: Mapping[str, str] | None) -> dict[str, str]:
    """``params`` without any entry whose name names a credential.

    Used to build the HTTP cache key. Everything not on the list survives,
    including names this project has never seen.
    """
    if not params:
        return {}
    return {name: value for name, value in params.items() if name.lower() not in CREDENTIAL_NAMES}


def _redacted_argument(value: object) -> object:
    """One ``%``-formatting argument, redacted if it renders to a credential.

    ``str(value)`` rather than an ``isinstance(value, str)`` guard, because the
    argument that carries the key is an ``httpx.URL``, not a string, and a type
    check misses it. The original object is returned when nothing changed, so an
    ``int`` formatted with ``%d`` stays an ``int``.
    """
    if isinstance(value, str):
        return redact_text(value)
    rendered = str(value)
    redacted = redact_text(rendered)
    return redacted if redacted != rendered else value


class CredentialRedactingFilter(logging.Filter):
    """Remove credential values from a log record, whatever produced it.

    Installed on the *handler*, not on a logger. A filter on the ``httpx`` logger
    would cover the one line known to leak today; a filter on the handler covers
    every logger routed through it, which includes libraries neither this module
    nor its author has thought of.

    Both halves of the record are rewritten. ``record.args`` is rewritten so the
    message formats clean wherever it is formatted, and ``record.msg`` is
    rewritten because a caller may have interpolated the credential itself before
    handing the string over. Rewriting only one leaves the other as the leak.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact_text(record.msg)

        args = record.args
        if isinstance(args, Mapping):
            record.args = {name: _redacted_argument(value) for name, value in args.items()}
        elif isinstance(args, tuple):
            record.args = tuple(_redacted_argument(value) for value in args)
        return True


def redact_processor(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
    """structlog processor doing what the handler filter cannot reach.

    ``configure_logging`` uses ``PrintLoggerFactory``, so a structlog event never
    becomes a ``LogRecord`` and never meets a handler. Without this the handler
    filter would be a component claiming coverage it does not have.

    Two cases: a bound key that names a credential, and a credential interpolated
    into a string value.
    """
    for key, value in list(event_dict.items()):
        if key.lower() in CREDENTIAL_NAMES:
            event_dict[key] = REDACTED
        elif isinstance(value, str):
            event_dict[key] = redact_text(value)
    return event_dict
