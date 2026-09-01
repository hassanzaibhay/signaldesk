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

import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Final

import httpx

from signaldesk.core.config import Settings, get_settings
from signaldesk.core.errors import IngestError
from signaldesk.core.http import CACHE_HEADER, get, is_cached
from signaldesk.core.logging import get_logger
from signaldesk.core.redaction import CREDENTIAL_NAMES

log = get_logger(__name__)

BASE_URL: Final[str] = "https://api.fda.gov/drug/label.json"

#: The query parameter carrying the credential.
#:
#: Checked against ``core.redaction`` at import, so renaming it without putting
#: the new name on the credential list stops the module loading rather than
#: quietly publishing the key in a log line and a cache key. That is a narrowing,
#: not a guarantee: a *second* credential added under a name nobody puts on the
#: list is still logged and still keys the cache.
API_KEY_PARAM: Final[str] = "api_key"

if API_KEY_PARAM not in CREDENTIAL_NAMES:  # pragma: no cover - import-time invariant
    _message = (
        f"{API_KEY_PARAM!r} is sent as a credential but is not in "
        "core.redaction.CREDENTIAL_NAMES, so it would be logged in full and "
        "would key the response cache"
    )
    raise IngestError(_message)

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
    """What one query returned, and what it cost to get it.

    Requests and cache hits are counted apart, as ``ingest.rxnorm.client`` does.
    A single total cannot answer the question the cost block exists for - what a
    re-run costs against openFDA - because a warm cache serves pages that never
    leave the machine. Bytes are split the same way and for the same reason:
    the old single ``bytes_received`` accumulated cached content too, so it
    measured parse volume and was read as transfer.
    """

    results: list[dict[str, Any]] = field(default_factory=list)
    #: Pages that actually left the machine.
    requests: int = 0
    #: Pages served from the on-disk cache. No network, no rate limiting.
    cache_hits: int = 0
    network_bytes: int = 0
    cache_bytes: int = 0
    #: True when the search stopped at ``max_pages`` with more to fetch, so the
    #: label set is truncated. Set where the cap is logged, so the flag and the
    #: log line cannot disagree.
    page_cap: bool = False

    @property
    def matched(self) -> bool:
        return bool(self.results)

    @property
    def pages(self) -> int:
        """Every page handled, cached or not."""
        return self.requests + self.cache_hits


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
        params[API_KEY_PARAM] = settings.openfda_api_key
    return params


def _from_cache(response: httpx.Response) -> bool:
    """Whether this response came off disk rather than the wire."""
    return bool(response.headers.get(CACHE_HEADER) == "hit")


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
    requests = 0
    cache_hits = 0
    network_bytes = 0
    cache_bytes = 0
    page_cap = False

    for page in range(max_pages):
        response = _fetch(search_expression, page * PAGE_LIMIT, settings)
        # The response says whether it was served from disk; a second is_cached
        # probe could race the store that just wrote it.
        if _from_cache(response):
            cache_hits += 1
            cache_bytes += len(response.content)
        else:
            requests += 1
            network_bytes += len(response.content)

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
        page_cap = True
        log.warning(
            "spl.search.page_cap", search=search_expression, pages=max_pages, kept=len(results)
        )

    return SearchResult(
        results=results,
        requests=requests,
        cache_hits=cache_hits,
        network_bytes=network_bytes,
        cache_bytes=cache_bytes,
        page_cap=page_cap,
    )


#: A query that returns results whenever openFDA is answering at all. The check
#: needs a known-200 request, and a 404 would be indistinguishable from the nine
#: that the real run expects.
PROBE_SEARCH: Final[str] = 'openfda.generic_name:"IBUPROFEN"'

#: The header api-umbrella attaches to a keyed response. Measured, not assumed:
#: across the 318 responses cached by the 10:16 and 06:05 runs, the 317 keyless
#: ones carry no rate-limit header at all and the one keyed page carries
#: ``x-ratelimit-limit: 240``. It discriminates by presence, not by value - 240 is
#: the per-minute ceiling either way, and the daily cap that does differ, 1,000
#: against 120,000, never appears in a header.
RATE_LIMIT_HEADER: Final[str] = "x-ratelimit-limit"
RATE_REMAINING_HEADER: Final[str] = "x-ratelimit-remaining"


@dataclass(frozen=True, slots=True)
class KeyProbe:
    """One request's evidence about the key, with nothing secret in it."""

    keyed: bool
    status: int
    results: int
    rate_limit: str | None
    rate_remaining: str | None


@dataclass(frozen=True, slots=True)
class KeyCheck:
    """Whether the key in this process works, and what the headers showed."""

    with_key: KeyProbe
    without_key: KeyProbe
    key_present: bool
    #: Where this ran. The whole point of the check is that it must run in the
    #: container that will run the ingest, since compose reads env_file at
    #: container creation and a host-side answer says nothing about the container.
    host: str

    @property
    def accepted(self) -> bool:
        """The load-bearing assertion: openFDA answered a keyed request with data.

        api-umbrella rejects an unknown key with 403 and an ``API_KEY_INVALID``
        body, so a 200 carrying results on a query known to match is proof the key
        was accepted. The rate-limit header is corroboration, not the proof.
        """
        return self.with_key.status == httpx.codes.OK and self.with_key.results > 0

    @property
    def header_discriminated(self) -> bool:
        """Whether presence of the rate-limit header told keyed from keyless."""
        return self.with_key.rate_limit is not None and self.without_key.rate_limit is None

    @property
    def inconclusive(self) -> bool:
        """Accepted, but the header did not appear where it was expected.

        Not a failure. A response served from the umbrella's own cache can omit
        the header on a keyed request, so its absence is a reason to run the check
        again rather than a verdict on the key.
        """
        return self.accepted and self.with_key.rate_limit is None


def _probe(settings: Settings, *, keyed: bool) -> KeyProbe:
    """One uncached request, keyed or not. Never logs the URL or the key."""
    params = {"search": PROBE_SEARCH, "limit": "1"}
    if keyed:
        params[API_KEY_PARAM] = settings.openfda_api_key
    response = get(BASE_URL, params=params, settings=settings, use_cache=False)
    results = 0
    if response.status_code == httpx.codes.OK:
        payload = response.json()
        results = len(payload.get("results") or [])
    return KeyProbe(
        keyed=keyed,
        status=response.status_code,
        results=results,
        rate_limit=response.headers.get(RATE_LIMIT_HEADER),
        rate_remaining=response.headers.get(RATE_REMAINING_HEADER),
    )


def verify_api_key(settings: Settings | None = None) -> KeyCheck:
    """Prove the key in this process is accepted by openFDA. Two requests, no cache.

    It exists because the run it precedes cannot prove this. A warm ``--force``
    run makes nine network requests and all nine are queries openFDA answers with
    404, so a rejected key would arrive as errors indistinguishable from the
    expected misses, and the artifact would record ``api_key_present: true`` for a
    run that never demonstrated the key works.

    Deliberately not wired into ``pipeline.run``. Gating the run on it would put a
    network call inside the run, so this stays a separate command, which means a
    human has to remember to run it. That is a weaker mechanism than the rest of
    this module and it is not presented as anything else.

    The cache is bypassed on both requests: a cached answer would prove only that
    somebody once had a working key.
    """
    settings = settings or get_settings()
    check = KeyCheck(
        with_key=_probe(settings, keyed=True),
        without_key=_probe(settings, keyed=False),
        key_present=bool(settings.openfda_api_key),
        host=socket.gethostname(),
    )
    log.info(
        "spl.key_check",
        host=check.host,
        key_present=check.key_present,
        accepted=check.accepted,
        keyed_status=check.with_key.status,
        keyless_status=check.without_key.status,
        keyed_rate_limit=check.with_key.rate_limit,
        keyless_rate_limit=check.without_key.rate_limit,
        header_discriminated=check.header_discriminated,
    )
    return check


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
