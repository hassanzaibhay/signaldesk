"""The openFDA client: paging, the 404 that means 'no match', and the limiter."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from signaldesk.core.config import Settings
from signaldesk.ingest.spl import client
from signaldesk.ingest.spl.client import SplRequestError, TokenBucket

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "spl"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(django_secret_key="test", data_dir=tmp_path, cache_dir=tmp_path / "cache")


@pytest.fixture
def keyed_settings(tmp_path: Path) -> Settings:
    return Settings(
        django_secret_key="test",
        data_dir=tmp_path,
        cache_dir=tmp_path / "cache",
        openfda_api_key="a-key",
    )


def _page(count: int, *, total: int, start: int = 0) -> dict[str, Any]:
    return {
        "meta": {"results": {"total": total}},
        "results": [
            {"set_id": f"set-{index}", "effective_time": "20240101"}
            for index in range(start, start + count)
        ],
    }


@respx.mock
def test_a_single_page_is_one_request(settings: Settings) -> None:
    respx.get(client.BASE_URL).mock(return_value=httpx.Response(200, json=_page(3, total=3)))
    result = client.search('openfda.brand_name:"X"', settings)
    assert result.matched
    assert len(result.results) == 3
    assert result.requests_made == 1


@respx.mock
def test_paging_continues_until_the_total_is_reached(settings: Settings) -> None:
    route = respx.get(client.BASE_URL)
    route.side_effect = [
        httpx.Response(200, json=_page(client.PAGE_LIMIT, total=150)),
        httpx.Response(200, json=_page(50, total=150, start=client.PAGE_LIMIT)),
    ]
    result = client.search('openfda.brand_name:"X"', settings)
    assert len(result.results) == 150
    assert result.requests_made == 2
    assert route.calls[1].request.url.params["skip"] == str(client.PAGE_LIMIT)


@respx.mock
def test_paging_is_capped(settings: Settings) -> None:
    """A term matching everything must not walk the whole corpus."""
    respx.get(client.BASE_URL).mock(
        return_value=httpx.Response(200, json=_page(client.PAGE_LIMIT, total=10_000))
    )
    result = client.search('openfda.generic_name:"WATER"', settings, max_pages=3)
    assert result.requests_made == 3
    assert len(result.results) == 3 * client.PAGE_LIMIT


@respx.mock
def test_a_404_is_no_match_not_a_failure(settings: Settings) -> None:
    """openFDA answers an empty search with 404. That outcome is the measurement."""
    body = json.loads((FIXTURES / "no_match.json").read_text(encoding="utf-8"))
    respx.get(client.BASE_URL).mock(return_value=httpx.Response(404, json=body))
    result = client.search('openfda.brand_name:"NOTADRUG"', settings)
    assert not result.matched
    assert result.results == []
    assert result.requests_made == 1


@respx.mock
def test_any_other_error_status_is_raised(settings: Settings) -> None:
    respx.get(client.BASE_URL).mock(return_value=httpx.Response(400, text="bad query"))
    with pytest.raises(SplRequestError, match="400"):
        client.search('openfda.brand_name:"["', settings)


@respx.mock
def test_both_name_fields_are_searched(settings: Settings) -> None:
    """A FAERS string is as often a generic name as a trade name."""
    route = respx.get(client.BASE_URL).mock(
        return_value=httpx.Response(200, json=_page(1, total=1))
    )
    client.by_brand_name("LIPITOR", settings)
    search = route.calls[0].request.url.params["search"]
    assert 'openfda.brand_name:"LIPITOR"' in search
    assert 'openfda.generic_name:"LIPITOR"' in search


@respx.mock
def test_the_api_key_is_sent_when_configured(keyed_settings: Settings) -> None:
    route = respx.get(client.BASE_URL).mock(
        return_value=httpx.Response(200, json=_page(1, total=1))
    )
    client.by_brand_name("LIPITOR", keyed_settings)
    assert route.calls[0].request.url.params["api_key"] == "a-key"


@respx.mock
def test_no_api_key_parameter_when_unset(settings: Settings) -> None:
    route = respx.get(client.BASE_URL).mock(
        return_value=httpx.Response(200, json=_page(1, total=1))
    )
    client.by_brand_name("LIPITOR", settings)
    assert "api_key" not in route.calls[0].request.url.params


@respx.mock
def test_a_repeat_query_is_served_from_the_cache(settings: Settings) -> None:
    """A re-run costs no requests, which is what makes widening the scope cheap."""
    route = respx.get(client.BASE_URL).mock(
        return_value=httpx.Response(200, json=_page(1, total=1))
    )
    client.by_brand_name("LIPITOR", settings)
    client.by_brand_name("LIPITOR", settings)
    assert route.call_count == 1


def test_the_daily_cap_depends_on_the_key(settings: Settings, keyed_settings: Settings) -> None:
    assert client.daily_cap(settings) == client.DAILY_CAP_WITHOUT_KEY
    assert client.daily_cap(keyed_settings) == client.DAILY_CAP_WITH_KEY


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('AB"CD', 'AB\\"CD'),
        ("AB\\CD", "AB\\\\CD"),
        ("PLAIN", "PLAIN"),
    ],
)
def test_quoting_cannot_break_out_of_the_search_phrase(raw: str, expected: str) -> None:
    assert client.quote(raw) == expected


def test_the_token_bucket_blocks_once_the_allowance_is_spent() -> None:
    bucket = TokenBucket(client.REQUESTS_PER_SECOND)
    for _ in range(int(client.REQUESTS_PER_SECOND)):
        bucket.take()

    started = time.monotonic()
    bucket.take()
    elapsed = time.monotonic() - started

    # The bucket started full, so the call after the allowance is the one that
    # waits. Half the nominal interval, because the clock this is measured with
    # is coarser than the interval on some platforms.
    assert elapsed >= 1.0 / client.REQUESTS_PER_SECOND * 0.5
    assert client.REQUESTS_PER_SECOND == 4.0
