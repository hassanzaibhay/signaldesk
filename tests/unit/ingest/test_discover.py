"""Quarter discovery: reading real hrefs rather than constructing them."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
import respx

from signaldesk.core.config import Settings
from signaldesk.core.errors import IngestError
from signaldesk.ingest.faers.discover import (
    EXTRACT_PAGE_URL,
    Discovery,
    cache_path,
    constructed_url,
    discover,
    parse_page,
    url_for,
)
from signaldesk.ingest.faers.quarter import Quarter

pytestmark = pytest.mark.unit

PAGE = """
<html><body>
  <a href="/content/Exports/faers_ascii_2013q1.zip">2013 Q1</a>
  <a href="https://fis.fda.gov/content/Exports/faers_ascii_2019Q1.zip">2019 Q1</a>
  <a href="/content/Exports/faers_ascii_2026q2.zip">2026 Q2</a>
  <a href="/content/Exports/aers_ascii_2011q1.zip">legacy, out of scope</a>
  <a href="/content/Exports/faers_xml_2013q1.zip">xml, not wanted</a>
  <a href="/some/other/page.html">unrelated</a>
</body></html>
"""


def test_parses_only_current_era_ascii_archives() -> None:
    urls = parse_page(PAGE)
    assert set(urls) == {Quarter(2013, 1), Quarter(2019, 1), Quarter(2026, 2)}


def test_relative_links_become_absolute() -> None:
    urls = parse_page(PAGE)
    assert urls[Quarter(2013, 1)].startswith("https://fis.fda.gov/")


def test_uppercase_quarter_letters_are_preserved() -> None:
    # A third of the corpus spells it this way; constructing the URL would 404.
    urls = parse_page(PAGE)
    assert urls[Quarter(2019, 1)].endswith("faers_ascii_2019Q1.zip")
    assert urls[Quarter(2019, 1)] != constructed_url(Quarter(2019, 1))


def test_url_for_prefers_the_discovered_link() -> None:
    discovery = Discovery(
        retrieved_at=datetime.now(tz=UTC), page_url=EXTRACT_PAGE_URL, urls=parse_page(PAGE)
    )
    assert url_for(Quarter(2019, 1), discovery).endswith("2019Q1.zip")


def test_url_for_falls_back_when_a_quarter_is_not_listed() -> None:
    discovery = Discovery(
        retrieved_at=datetime.now(tz=UTC), page_url=EXTRACT_PAGE_URL, urls=parse_page(PAGE)
    )
    assert url_for(Quarter(2014, 3), discovery) == constructed_url(Quarter(2014, 3))


def test_latest_is_the_newest_quarter() -> None:
    discovery = Discovery(
        retrieved_at=datetime.now(tz=UTC), page_url=EXTRACT_PAGE_URL, urls=parse_page(PAGE)
    )
    assert discovery.latest == Quarter(2026, 2)
    assert discovery.quarters[0] == Quarter(2013, 1)


def test_latest_on_an_empty_discovery_raises() -> None:
    discovery = Discovery(retrieved_at=datetime.now(tz=UTC), page_url=EXTRACT_PAGE_URL, urls={})
    with pytest.raises(IngestError, match="no quarterly archives"):
        _ = discovery.latest


@respx.mock
def test_discover_fetches_and_caches(settings: Settings) -> None:
    route = respx.get(EXTRACT_PAGE_URL).mock(return_value=httpx.Response(200, text=PAGE))

    first = discover(settings, refresh=True)
    assert route.call_count == 1
    assert cache_path(settings).is_file()

    second = discover(settings)
    assert route.call_count == 1, "a fresh cache must not hit the network"
    assert second.urls == first.urls


@respx.mock
def test_stale_cache_is_refetched(settings: Settings) -> None:
    respx.get(EXTRACT_PAGE_URL).mock(return_value=httpx.Response(200, text=PAGE))
    discover(settings, refresh=True)

    reloaded = discover(settings, max_age=timedelta(seconds=0))
    assert reloaded.urls


@respx.mock
def test_unreadable_cache_is_treated_as_a_miss(settings: Settings) -> None:
    route = respx.get(EXTRACT_PAGE_URL).mock(return_value=httpx.Response(200, text=PAGE))
    path = cache_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ not json", encoding="utf-8")

    assert discover(settings).urls
    assert route.call_count == 1


@respx.mock
def test_a_page_with_no_archives_raises(settings: Settings) -> None:
    respx.get(EXTRACT_PAGE_URL).mock(return_value=httpx.Response(200, text="<html></html>"))
    with pytest.raises(IngestError, match="listed no quarterly archives"):
        discover(settings, refresh=True)


@respx.mock
def test_a_failed_request_raises(settings: Settings) -> None:
    respx.get(EXTRACT_PAGE_URL).mock(return_value=httpx.Response(503))
    with pytest.raises(IngestError, match="returned 503"):
        discover(settings, refresh=True)


def test_cache_path_sits_under_the_cache_directory(settings: Settings, tmp_path: Path) -> None:
    assert cache_path(settings).parent.name == "faers"
