"""The openFDA drug label endpoint.

Everything goes through ``core.http``, so retries, timeouts, the user agent and
the on-disk response cache are the shared ones and a re-run costs no requests.

Two properties of this API shape the module:

* **404 means no matches, not failure.** openFDA answers a search that matched
  nothing with 404 and an error body. Treating that as an error would turn the
  normal "this drug has no label under that name" outcome into a failed ingest
  unit, and the hit rate this pipeline exists to measure is exactly the rate of
  those 404s. It is returned as an empty result.
* **The daily cap binds, not the rate.** 240 requests per minute either way, but
  1,000 per day without an API key against 120,000 with one. The key is free and
  is read from ``OPENFDA_API_KEY``. Without it a scope above a thousand strings
  is a multi-day operation, so the absence of a key is logged loudly at the
  start of a run rather than discovered when the quota runs out.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Final

import httpx

from signaldesk.core.config import Settings, get_settings
from signaldesk.core.errors import IngestError
from signaldesk.core.http import get, is_cached
from signaldesk.core.logging import get_logger

log = get_logger(__name__)

BASE_URL: Final[str] = "https://api.fda.gov/drug/label.json"

#: The published ceiling, 240 per minute. Not a tuning knob.
REQUESTS_PER_SECOND: Final[float] = 4.0

#: openFDA's maximum page size for this endpoint.
PAGE_LIMIT: Final[int] = 100

#: Pages per query. A drug string matching more than a thousand labels is a
#: generic term rather than a product, and the pages past this add duplicates of
#: the same sections rather than new information.
MAX_PAGES: Final[int] = 10

#: Daily request cap, with and without a key. Reported, not enforced: the
#: service enforces it, and this is here so the log can say what was assumed.
DAILY_CAP_WITH_KEY: Final[int] = 120_000
DAILY_CAP_WITHOUT_KEY: Final[int] = 1_000


class SplRequestError(IngestError):
    """openFDA answered with something that is neither results nor 'no match'."""


class TokenBucket:
    """A process-global rate limiter, since the published limit is per IP."""

    def __init__(self, per_second: float) -> None:
        self._per_second = per_second
        self._lock = threading.Lock()
        self._allowance = per_second
        self._checked_at = time.monotonic()

    def take(self) -> None:
        """Block until one request may be sent."""
        with self._lock:
            now = time.monotonic()
            self._allowance = min(
                self._per_second,
                self._allowance + (now - self._checked_at) * self._per_second,
            )
            self._checked_at = now
            if self._allowance < 1.0:
                delay = (1.0 - self._allowance) / self._per_second
                time.sleep(delay)
                self._checked_at = time.monotonic()
                self._allowance = 0.0
            else:
                self._allowance -= 1.0


_LIMITER = TokenBucket(REQUESTS_PER_SECOND)


@dataclass(frozen=True, slots=True)
class SearchResult:
    """What one query returned."""

    results: list[dict[str, Any]] = field(default_factory=list)
    requests_made: int = 0
    bytes_received: int = 0

    @property
    def matched(self) -> bool:
        return bool(self.results)


def quote(value: str) -> str:
    """Escape a value for an openFDA ``search`` phrase.

    Lucene-ish syntax: the phrase is wrapped in double quotes, so an embedded
    quote or backslash would end the phrase early and change which field is
    being searched.
    """
    return value.replace("\\", "\\\\").replace('"', '\\"')


def daily_cap(settings: Settings | None = None) -> int:
    """The request budget a run may assume today."""
    settings = settings or get_settings()
    return DAILY_CAP_WITH_KEY if settings.openfda_api_key else DAILY_CAP_WITHOUT_KEY


def _params(search: str, skip: int, settings: Settings) -> dict[str, str]:
    params = {"search": search, "limit": str(PAGE_LIMIT), "skip": str(skip)}
    if settings.openfda_api_key:
        params["api_key"] = settings.openfda_api_key
    return params


def _fetch(search: str, skip: int, settings: Settings) -> httpx.Response:
    """One page, rate limited only when it will actually leave the machine.

    The cache check comes before the token bucket. A limiter in front of every
    call throttles cache hits too, which turns a warm re-run into minutes of
    sleeping for requests that are never sent.
    """
    params = _params(search, skip, settings)
    if not is_cached(BASE_URL, params=params, settings=settings):
        _LIMITER.take()
    return get(BASE_URL, params=params, settings=settings)


def search(
    search_expression: str, settings: Settings | None = None, *, max_pages: int = MAX_PAGES
) -> SearchResult:
    """Run one openFDA search to exhaustion, or to ``max_pages``."""
    settings = settings or get_settings()
    results: list[dict[str, Any]] = []
    requests_made = 0
    bytes_received = 0

    for page in range(max_pages):
        response = _fetch(search_expression, page * PAGE_LIMIT, settings)
        requests_made += 1
        bytes_received += len(response.content)

        if response.status_code == httpx.codes.NOT_FOUND:
            # No match. Normal, and the thing being measured.
            log.debug("spl.search.no_match", search=search_expression, page=page)
            break
        if response.status_code != httpx.codes.OK:
            message = (
                f"openFDA returned {response.status_code} for {search_expression!r}; "
                f"body starts {response.text[:200]!r}"
            )
            raise SplRequestError(message)

        payload = response.json()
        batch = payload.get("results") or []
        results.extend(batch)
        total = int(payload.get("meta", {}).get("results", {}).get("total", len(results)))
        if len(batch) < PAGE_LIMIT or len(results) >= total:
            break
    else:
        log.warning(
            "spl.search.page_cap", search=search_expression, pages=max_pages, kept=len(results)
        )

    return SearchResult(results=results, requests_made=requests_made, bytes_received=bytes_received)


def by_brand_name(name: str, settings: Settings | None = None) -> SearchResult:
    """Labels whose brand or generic name matches ``name``.

    Both fields, because a FAERS string is as often a generic name as a trade
    name and querying only ``brand_name`` misses every generic report.
    """
    escaped = quote(name)
    expression = f'openfda.brand_name:"{escaped}" OR openfda.generic_name:"{escaped}"'
    return search(expression, settings)


def by_rxcui(rxcui: int, settings: Settings | None = None) -> SearchResult:
    """Labels openFDA associates with an RxNorm concept.

    Correct, and currently unreachable: normalization produces no rxcuis on this
    machine. Kept so the ingredient route works the day the mapping is rebuilt.
    """
    return search(f'openfda.rxcui:"{quote(str(rxcui))}"', settings)
