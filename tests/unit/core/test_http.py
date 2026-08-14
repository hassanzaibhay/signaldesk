"""HTTP client behaviour: identity, retries, Retry-After, and the disk cache.

Everything downstream depends on this module behaving under a flaky endpoint, so
the retry and cache paths are exercised rather than assumed. Sleeping is
injected as a no-op: the tests assert the schedule, not the wall clock.
"""

from __future__ import annotations

import gzip
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import httpx
import pytest
import respx

from signaldesk.core import http
from signaldesk.core.config import Settings

pytestmark = pytest.mark.unit

URL = "https://example.test/resource"


@pytest.fixture
def slept() -> list[float]:
    return []


@pytest.fixture
def no_sleep(slept: list[float]) -> Callable[[float], None]:
    def record(seconds: float) -> None:
        slept.append(seconds)

    return record


@respx.mock
def test_user_agent_identifies_the_project(settings: Settings) -> None:
    route = respx.get(URL).mock(return_value=httpx.Response(200, text="ok"))
    http.get(URL, settings=settings, use_cache=False)
    assert route.calls.last.request.headers["User-Agent"] == settings.http_user_agent


@respx.mock
def test_client_sets_explicit_timeouts(settings: Settings) -> None:
    with http.build_client(settings) as client:
        assert client.timeout.connect == http.CONNECT_TIMEOUT_SECONDS
        assert client.timeout.read == http.READ_TIMEOUT_SECONDS


@respx.mock
def test_retries_a_server_error_then_succeeds(settings: Settings, no_sleep) -> None:
    route = respx.get(URL).mock(
        side_effect=[httpx.Response(503), httpx.Response(200, text="second time")]
    )
    response = http.get(URL, settings=settings, use_cache=False, sleep=no_sleep)
    assert response.status_code == 200
    assert response.text == "second time"
    assert route.call_count == 2


@respx.mock
def test_gives_up_after_max_attempts_and_returns_the_last_response(
    settings: Settings, no_sleep
) -> None:
    route = respx.get(URL).mock(return_value=httpx.Response(503))
    response = http.get(URL, settings=settings, use_cache=False, sleep=no_sleep)
    assert response.status_code == 503
    assert route.call_count == http.MAX_ATTEMPTS


@respx.mock
def test_does_not_retry_a_client_error(settings: Settings, no_sleep) -> None:
    route = respx.get(URL).mock(return_value=httpx.Response(404))
    response = http.get(URL, settings=settings, use_cache=False, sleep=no_sleep)
    assert response.status_code == 404
    assert route.call_count == 1


@respx.mock
def test_transport_errors_are_retried_and_finally_raised(settings: Settings, no_sleep) -> None:
    route = respx.get(URL).mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(httpx.ConnectError):
        http.get(URL, settings=settings, use_cache=False, sleep=no_sleep)
    assert route.call_count == http.MAX_ATTEMPTS


@respx.mock
def test_rate_limit_delay_is_at_least_the_requested_wait(
    settings: Settings, no_sleep, slept: list[float]
) -> None:
    respx.get(URL).mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "7"}),
            httpx.Response(200, text="ok"),
        ]
    )
    response = http.get(URL, settings=settings, use_cache=False, sleep=no_sleep)
    assert response.status_code == 200
    assert slept == [pytest.approx(7.0)]


def test_retry_after_accepts_a_delay_in_seconds() -> None:
    response = httpx.Response(429, headers={"Retry-After": "12"})
    assert http.retry_after_seconds(response) == pytest.approx(12.0)


def test_retry_after_accepts_an_http_date() -> None:
    when = datetime.now(tz=UTC) + timedelta(seconds=30)
    response = httpx.Response(503, headers={"Retry-After": format_datetime(when)})
    seconds = http.retry_after_seconds(response)
    assert seconds is not None
    assert 25 <= seconds <= 31


def test_retry_after_is_absent_when_the_header_is_missing() -> None:
    assert http.retry_after_seconds(httpx.Response(503)) is None


def test_unparseable_retry_after_is_ignored() -> None:
    response = httpx.Response(503, headers={"Retry-After": "soon"})
    assert http.retry_after_seconds(response) is None


def test_retry_after_is_capped(settings: Settings, no_sleep, slept: list[float]) -> None:
    with respx.mock:
        respx.get(URL).mock(
            side_effect=[
                httpx.Response(429, headers={"Retry-After": "3600"}),
                httpx.Response(200),
            ]
        )
        http.get(URL, settings=settings, use_cache=False, sleep=no_sleep)
    assert slept == [pytest.approx(http.MAX_RETRY_AFTER_SECONDS)]


@respx.mock
def test_a_second_get_is_served_from_disk(settings: Settings) -> None:
    route = respx.get(URL).mock(return_value=httpx.Response(200, text="cached body"))

    first = http.get(URL, settings=settings)
    second = http.get(URL, settings=settings)

    assert route.call_count == 1
    assert first.text == second.text == "cached body"
    assert second.headers[http.CACHE_HEADER] == "hit"


@respx.mock
def test_error_responses_are_not_cached(settings: Settings, no_sleep) -> None:
    respx.get(URL).mock(return_value=httpx.Response(500))
    http.get(URL, settings=settings, sleep=no_sleep)

    key = http.ResponseCache.key("GET", URL, None)
    assert http.ResponseCache(settings.http_cache_dir).load(key) is None


@respx.mock
def test_query_parameters_are_part_of_the_cache_key(settings: Settings) -> None:
    route = respx.get(URL).mock(return_value=httpx.Response(200, text="ok"))

    http.get(URL, settings=settings, params={"page": "1"})
    http.get(URL, settings=settings, params={"page": "2"})

    assert route.call_count == 2


def test_cache_key_ignores_parameter_order() -> None:
    left = http.ResponseCache.key("GET", URL, {"a": "1", "b": "2"})
    right = http.ResponseCache.key("get", URL, {"b": "2", "a": "1"})
    assert left == right


def test_corrupt_cache_entry_is_treated_as_a_miss(settings: Settings) -> None:
    cache = http.ResponseCache(settings.http_cache_dir)
    key = http.ResponseCache.key("GET", URL, None)
    path = cache.path_for(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ not json", encoding="utf-8")
    assert cache.load(key) is None


@respx.mock
def test_a_compressed_response_survives_the_cache(settings: Settings) -> None:
    """The cache stores decoded bytes, so it must not keep the encoding header.

    RxNav compresses. Storing `Content-Encoding: gzip` beside an already-decoded
    body made the replay try to gunzip plain JSON and raise `DecodingError`, so
    the second pass over half a million lookups failed rather than being free.
    The FAERS host is asked for `identity` and mocked responses carry no encoding
    header, which is why nothing caught it until a real compressed endpoint.
    """
    body = b'{"idGroup": {"rxnormId": ["6851"]}}'
    route = respx.get(URL).mock(
        return_value=httpx.Response(
            200,
            content=gzip.compress(body),
            headers={"Content-Encoding": "gzip", "Content-Type": "application/json"},
        )
    )

    first = http.get(URL, settings=settings)
    second = http.get(URL, settings=settings)

    assert route.call_count == 1
    assert second.headers[http.CACHE_HEADER] == "hit"
    assert second.content == first.content == body
    assert second.json() == {"idGroup": {"rxnormId": ["6851"]}}
    assert "content-encoding" not in second.headers
