"""The one HTTP client.

Every external call in this project goes through here. Centralising it buys four
things that matter when the pipelines are re-run repeatedly against public data
services:

* explicit connect and read timeouts, so a hung endpoint cannot stall an ingest;
* bounded retries with exponential backoff and jitter, honouring ``Retry-After``
  when a service asks for a specific delay;
* an on-disk response cache, so a re-run costs nothing and works offline;
* a descriptive ``User-Agent`` identifying the project, which is a condition of
  use for several of the data sources.

No other module constructs a client.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Final

import httpx
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception_type,
    retry_if_result,
    stop_after_attempt,
    wait_exponential_jitter,
)
from tenacity.nap import sleep as default_sleep

from signaldesk.core.config import Settings, get_settings
from signaldesk.core.logging import get_logger

log = get_logger(__name__)

CONNECT_TIMEOUT_SECONDS: Final[float] = 10.0
READ_TIMEOUT_SECONDS: Final[float] = 30.0
WRITE_TIMEOUT_SECONDS: Final[float] = 10.0
POOL_TIMEOUT_SECONDS: Final[float] = 5.0

MAX_ATTEMPTS: Final[int] = 3
BACKOFF_INITIAL_SECONDS: Final[float] = 0.5
BACKOFF_MAX_SECONDS: Final[float] = 8.0
BACKOFF_JITTER_SECONDS: Final[float] = 0.5

#: Statuses worth trying again: the service is busy, throttling, or briefly broken.
RETRY_STATUS_CODES: Final[frozenset[int]] = frozenset({408, 425, 429, 500, 502, 503, 504})

#: Cap on an obeyed ``Retry-After``. A service asking for an hour is telling us to
#: stop for now, not to hold a connection open for an hour.
MAX_RETRY_AFTER_SECONDS: Final[float] = 60.0

CACHE_HEADER: Final[str] = "x-signaldesk-cache"


def build_client(settings: Settings | None = None, *, base_url: str = "") -> httpx.Client:
    """Create the shared client. Callers own closing it, or use ``request``."""
    settings = settings or get_settings()
    timeout = httpx.Timeout(
        connect=CONNECT_TIMEOUT_SECONDS,
        read=READ_TIMEOUT_SECONDS,
        write=WRITE_TIMEOUT_SECONDS,
        pool=POOL_TIMEOUT_SECONDS,
    )
    return httpx.Client(
        base_url=base_url,
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": settings.http_user_agent},
        limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
    )


class ResponseCache:
    """A content-addressed cache of successful GET responses on disk.

    Keyed on method, URL and query parameters. Only 200 responses are stored:
    caching an error would turn a transient outage into a permanent one.
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    @staticmethod
    def key(method: str, url: str, params: Mapping[str, str] | None = None) -> str:
        canonical = json.dumps(
            {
                "method": method.upper(),
                "url": url,
                "params": dict(sorted(params.items())) if params else {},
            },
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def path_for(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def load(self, key: str) -> httpx.Response | None:
        path = self.path_for(key)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        headers = dict(payload["headers"])
        headers[CACHE_HEADER] = "hit"
        return httpx.Response(
            status_code=int(payload["status"]),
            headers=headers,
            content=base64.b64decode(payload["content"]),
        )

    def store(self, key: str, response: httpx.Response) -> None:
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": response.status_code,
            "headers": dict(response.headers),
            "content": base64.b64encode(response.content).decode("ascii"),
        }
        path.write_text(json.dumps(payload), encoding="utf-8")


def retry_after_seconds(response: httpx.Response) -> float | None:
    """Parse ``Retry-After``, which is either a delay in seconds or an HTTP date."""
    raw = response.headers.get("Retry-After")
    if raw is None:
        return None
    raw = raw.strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    delay: float = (when - datetime.now(tz=UTC)).total_seconds()
    return max(0.0, delay)


def _should_retry_response(response: httpx.Response) -> bool:
    return response.status_code in RETRY_STATUS_CODES


def _wait(state: RetryCallState) -> float:
    """Exponential backoff with jitter, raised to ``Retry-After`` when present."""
    backoff = wait_exponential_jitter(
        initial=BACKOFF_INITIAL_SECONDS,
        max=BACKOFF_MAX_SECONDS,
        jitter=BACKOFF_JITTER_SECONDS,
    )(state)
    outcome = state.outcome
    if outcome is None or outcome.failed:
        return backoff
    requested = retry_after_seconds(outcome.result())
    if requested is None:
        return backoff
    return min(max(backoff, requested), MAX_RETRY_AFTER_SECONDS)


def _return_last_outcome(state: RetryCallState) -> httpx.Response:
    """After the final attempt, hand back the last response (or re-raise)."""
    outcome = state.outcome
    if outcome is None:  # pragma: no cover - tenacity always sets an outcome here
        message = "retry finished without an outcome"
        raise RuntimeError(message)
    result: httpx.Response = outcome.result()
    return result


def request(
    method: str,
    url: str,
    *,
    client: httpx.Client | None = None,
    params: Mapping[str, str] | None = None,
    headers: Mapping[str, str] | None = None,
    settings: Settings | None = None,
    use_cache: bool = True,
    sleep: Callable[[float], None] | None = None,
) -> httpx.Response:
    """Perform an HTTP request with retries and, for GETs, the on-disk cache.

    Returns the response even when the final attempt failed with a retryable
    status: deciding what a 503 means belongs to the caller, which knows what it
    asked for. Transport failures propagate once the attempts are exhausted.

    ``sleep`` overrides the delay between attempts; tests pass a no-op so the
    backoff behaviour can be exercised without real waiting.
    """
    settings = settings or get_settings()
    cacheable = use_cache and method.upper() == "GET"
    cache = ResponseCache(settings.http_cache_dir)
    key = ResponseCache.key(method, url, params)

    if cacheable:
        cached = cache.load(key)
        if cached is not None:
            log.debug("http.cache.hit", method=method.upper(), url=url)
            return cached

    owns_client = client is None
    active = client or build_client(settings)
    try:
        retrying = Retrying(
            stop=stop_after_attempt(MAX_ATTEMPTS),
            wait=_wait,
            retry=(
                retry_if_exception_type(httpx.TransportError)
                | retry_if_result(_should_retry_response)
            ),
            retry_error_callback=_return_last_outcome,
            sleep=sleep if sleep is not None else default_sleep,
        )
        response = retrying(
            _send,
            active,
            method,
            url,
            params,
            headers,
        )
    finally:
        if owns_client:
            active.close()

    if cacheable and response.status_code == httpx.codes.OK:
        cache.store(key, response)
    return response


def _send(
    client: httpx.Client,
    method: str,
    url: str,
    params: Mapping[str, str] | None,
    headers: Mapping[str, str] | None,
) -> httpx.Response:
    response = client.request(method, url, params=params, headers=headers)
    log.debug("http.request", method=method.upper(), url=url, status=response.status_code)
    return response


def get(
    url: str,
    *,
    client: httpx.Client | None = None,
    params: Mapping[str, str] | None = None,
    headers: Mapping[str, str] | None = None,
    settings: Settings | None = None,
    use_cache: bool = True,
    sleep: Callable[[float], None] | None = None,
) -> httpx.Response:
    """GET ``url`` through the shared client, cache and retry policy."""
    return request(
        "GET",
        url,
        client=client,
        params=params,
        headers=headers,
        settings=settings,
        use_cache=use_cache,
        sleep=sleep,
    )
