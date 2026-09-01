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
    # openfda_api_key is pinned empty rather than left to the environment.
    # _env_file=None blocks the .env file but not the process environment, so
    # without this the cap and api_key assertions below measure whichever
    # machine runs them: they passed only while the container had no key, and
    # went red the moment one was supplied.
    return Settings(
        _env_file=None,
        django_secret_key="test",
        data_dir=tmp_path,
        cache_dir=tmp_path / "cache",
        openfda_api_key="",
    )


@pytest.fixture
def keyed_settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
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
    assert result.requests == 1


@respx.mock
def test_paging_continues_until_the_total_is_reached(settings: Settings) -> None:
    route = respx.get(client.BASE_URL)
    route.side_effect = [
        httpx.Response(200, json=_page(client.PAGE_LIMIT, total=150)),
        httpx.Response(200, json=_page(50, total=150, start=client.PAGE_LIMIT)),
    ]
    result = client.search('openfda.brand_name:"X"', settings)
    assert len(result.results) == 150
    assert result.requests == 2
    assert route.calls[1].request.url.params["skip"] == str(client.PAGE_LIMIT)


@respx.mock
def test_paging_is_capped(settings: Settings) -> None:
    """A term matching everything must not walk the whole corpus."""
    respx.get(client.BASE_URL).mock(
        return_value=httpx.Response(200, json=_page(client.PAGE_LIMIT, total=10_000))
    )
    result = client.search('openfda.generic_name:"WATER"', settings, max_pages=3)
    assert result.requests == 3
    assert len(result.results) == 3 * client.PAGE_LIMIT


@respx.mock
def test_a_404_is_no_match_not_a_failure(settings: Settings) -> None:
    """openFDA answers an empty search with 404. That outcome is the measurement."""
    body = json.loads((FIXTURES / "no_match.json").read_text(encoding="utf-8"))
    respx.get(client.BASE_URL).mock(return_value=httpx.Response(404, json=body))
    result = client.search('openfda.brand_name:"NOTADRUG"', settings)
    assert not result.matched
    assert result.results == []
    assert result.requests == 1


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


@respx.mock
def test_a_cached_page_is_a_cache_hit_and_not_a_request(settings: Settings) -> None:
    """cost.requests counted cache hits, so a warm re-run read as network cost.

    The second call is served off disk by the shared client and never reaches
    respx, so the route's call count is the proof that nothing left the machine.
    """
    route = respx.get(client.BASE_URL).mock(
        return_value=httpx.Response(200, json=_page(3, total=3))
    )
    cold = client.search('openfda.brand_name:"X"', settings)
    warm = client.search('openfda.brand_name:"X"', settings)

    assert route.call_count == 1, "the second search must not reach the network"
    assert (cold.requests, cold.cache_hits) == (1, 0)
    assert (warm.requests, warm.cache_hits) == (0, 1)
    assert cold.pages == warm.pages == 1
    assert warm.results == cold.results


@respx.mock
def test_cached_bytes_are_not_counted_as_transfer(settings: Settings) -> None:
    """bytes_received summed cached content too, so it measured parse volume."""
    respx.get(client.BASE_URL).mock(return_value=httpx.Response(200, json=_page(3, total=3)))
    cold = client.search('openfda.brand_name:"X"', settings)
    warm = client.search('openfda.brand_name:"X"', settings)

    assert cold.network_bytes > 0
    assert cold.cache_bytes == 0
    assert warm.network_bytes == 0
    assert warm.cache_bytes > 0


@respx.mock
def test_the_page_cap_is_recorded_on_the_result_not_only_logged(settings: Settings) -> None:
    """A truncated label set that lives in a log line is not traceable."""
    respx.get(client.BASE_URL).mock(
        return_value=httpx.Response(200, json=_page(client.PAGE_LIMIT, total=10_000))
    )
    result = client.search('openfda.generic_name:"WATER"', settings, max_pages=3)
    assert result.page_cap is True
    assert result.requests == 3


@respx.mock
def test_a_search_that_ends_naturally_does_not_flag_the_cap(settings: Settings) -> None:
    respx.get(client.BASE_URL).mock(return_value=httpx.Response(200, json=_page(3, total=3)))
    result = client.search('openfda.brand_name:"X"', settings)
    assert result.page_cap is False


class TestTheKeySmokeCheck:
    """Proving the key works, because the run it precedes cannot.

    A warm --force run makes nine network requests and all nine are queries
    openFDA answers with 404. A rejected key would arrive as errors that look
    exactly like the expected misses, and the artifact would record
    api_key_present true for a run that never demonstrated the key works.
    """

    def _ok(self, *, rate_limit: str | None = "240") -> httpx.Response:
        headers = {}
        if rate_limit is not None:
            headers = {"x-ratelimit-limit": rate_limit, "x-ratelimit-remaining": "239"}
        return httpx.Response(200, json=_page(1, total=1), headers=headers)

    @respx.mock
    def test_a_keyed_200_with_results_is_the_proof(self, keyed_settings: Settings) -> None:
        """Status and payload, not the header, are what carry the verdict."""
        respx.get(client.BASE_URL).mock(
            side_effect=[self._ok(), self._ok(rate_limit=None)],
        )
        check = client.verify_api_key(keyed_settings)

        assert check.accepted
        assert check.key_present
        assert check.with_key.results == 1
        assert not check.inconclusive
        assert check.host

    @respx.mock
    def test_the_header_discriminates_by_presence_not_by_value(
        self, keyed_settings: Settings
    ) -> None:
        """Measured against the cached corpus, and re-confirmed here.

        Across the 318 responses on disk, the 317 keyless ones carry no
        rate-limit header and the one keyed page carries 240. The value is the
        per-minute ceiling and is the same either way; the daily cap that does
        differ, 1,000 against 120,000, never appears in a header. So the check
        must not assert on the value.
        """
        respx.get(client.BASE_URL).mock(
            side_effect=[self._ok(rate_limit="240"), self._ok(rate_limit=None)],
        )
        check = client.verify_api_key(keyed_settings)

        assert check.header_discriminated
        assert check.with_key.rate_limit == "240"
        assert check.without_key.rate_limit is None

    @respx.mock
    def test_a_rejected_key_fails(self, keyed_settings: Settings) -> None:
        """api-umbrella answers an unknown key with 403, not with a 404."""
        respx.get(client.BASE_URL).mock(
            side_effect=[
                httpx.Response(403, json={"error": {"code": "API_KEY_INVALID"}}),
                self._ok(rate_limit=None),
            ],
        )
        check = client.verify_api_key(keyed_settings)

        assert not check.accepted
        assert check.with_key.status == 403

    @respx.mock
    def test_an_absent_header_on_an_accepted_key_is_inconclusive_not_a_pass(
        self, keyed_settings: Settings
    ) -> None:
        """A response served from the umbrella's own cache can omit the header.

        Reporting that as a clean pass would assert the header discriminated
        when it did not; reporting it as a failure would blame a key that
        openFDA just accepted. It is a reason to run the check again.
        """
        respx.get(client.BASE_URL).mock(
            side_effect=[self._ok(rate_limit=None), self._ok(rate_limit=None)],
        )
        check = client.verify_api_key(keyed_settings)

        assert check.accepted
        assert check.inconclusive
        assert not check.header_discriminated

    @respx.mock
    def test_neither_request_touches_the_cache(self, keyed_settings: Settings) -> None:
        """A cached answer would prove only that somebody once had a working key."""
        route = respx.get(client.BASE_URL).mock(
            side_effect=[self._ok(), self._ok(rate_limit=None), self._ok(), self._ok()],
        )
        client.verify_api_key(keyed_settings)
        client.verify_api_key(keyed_settings)

        assert route.call_count == 4

    @respx.mock
    def test_the_key_is_never_rendered(self, keyed_settings: Settings) -> None:
        """The check reports headers and a boolean, never the credential.

        The KeyProbe carries no url and no key field, so there is nothing for a
        caller to print by accident. Asserted on the object rather than on the
        command line's output, because the command line formats what this gives
        it.
        """
        respx.get(client.BASE_URL).mock(
            side_effect=[self._ok(), self._ok(rate_limit=None)],
        )
        check = client.verify_api_key(keyed_settings)

        rendered = repr(check)
        assert keyed_settings.openfda_api_key not in rendered
        assert "api.fda.gov" not in rendered


def test_the_credential_parameter_is_on_the_redaction_list() -> None:
    """Renaming it without listing it must not be possible quietly.

    The import-time check in the client is what enforces this; the test states
    the invariant so the check itself cannot be deleted as unused.
    """
    from signaldesk.core.redaction import CREDENTIAL_NAMES

    assert client.API_KEY_PARAM in CREDENTIAL_NAMES
