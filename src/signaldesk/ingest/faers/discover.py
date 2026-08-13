"""Discover which quarterly extracts exist, and where they actually live.

The archive URLs are not constructible. Most quarters render the quarter letter
in lower case, but at least one (2025Q4, observed 2026-08-12) uses upper case, so
code that formats a URL from a quarter gets a 404 on that file and on any future
file whose name is spelled differently again.

So: read the page, keep the href that is really there, and cache the result. The
constructed form remains as a fallback for a quarter the page does not list, and
says so in the log when it is used.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin

from signaldesk.core.config import Settings, get_settings
from signaldesk.core.errors import IngestError
from signaldesk.core.http import get
from signaldesk.core.logging import get_logger
from signaldesk.ingest.faers.quarter import Quarter

log = get_logger(__name__)

EXTRACT_PAGE_URL = "https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html"

#: Current-era archives only. Legacy AERS (pre-2012Q4) uses an "aers_ascii_"
#: prefix and an incompatible schema, and is out of scope.
ARCHIVE_NAME = re.compile(r"faers_ascii_(?P<year>\d{4})[Qq](?P<quarter>[1-4])\.zip$")

CACHE_MAX_AGE = timedelta(days=7)


class _AnchorCollector(HTMLParser):
    """Collect every href on the page. Stdlib, so no parser dependency."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        for name, value in attrs:
            if name == "href" and value:
                self.hrefs.append(value)


@dataclass(frozen=True, slots=True)
class Discovery:
    """What the extract page listed, and when."""

    retrieved_at: datetime
    page_url: str
    urls: dict[Quarter, str]

    @property
    def quarters(self) -> list[Quarter]:
        return sorted(self.urls)

    @property
    def latest(self) -> Quarter:
        if not self.urls:
            message = "no quarterly archives were found on the extract page"
            raise IngestError(message)
        return max(self.urls)


def cache_path(settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    return settings.cache_dir / "faers" / "quarters.json"


def constructed_url(quarter: Quarter) -> str:
    """The usual spelling, for a quarter the page did not list."""
    return f"https://fis.fda.gov/content/Exports/faers_ascii_{quarter.year}q{quarter.quarter}.zip"


def parse_page(html: str, page_url: str = EXTRACT_PAGE_URL) -> dict[Quarter, str]:
    """Extract quarter to absolute URL from the page markup."""
    collector = _AnchorCollector()
    collector.feed(html)

    urls: dict[Quarter, str] = {}
    for href in collector.hrefs:
        match = ARCHIVE_NAME.search(href)
        if match is None:
            continue
        quarter = Quarter(year=int(match["year"]), quarter=int(match["quarter"]))
        absolute = urljoin(page_url, href)
        previous = urls.get(quarter)
        if previous is not None and previous != absolute:
            log.warning(
                "faers.discover.duplicate_quarter",
                quarter=str(quarter),
                kept=previous,
                ignored=absolute,
            )
            continue
        urls[quarter] = absolute
    return urls


def _write_cache(discovery: Discovery, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "retrieved_at": discovery.retrieved_at.isoformat(),
        "page_url": discovery.page_url,
        "quarters": {str(quarter): url for quarter, url in sorted(discovery.urls.items())},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_cache(path: Path) -> Discovery | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        retrieved_at = datetime.fromisoformat(payload["retrieved_at"])
        urls = {Quarter.parse(key): value for key, value in payload["quarters"].items()}
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        log.warning("faers.discover.cache_unreadable", error=type(exc).__name__)
        return None
    return Discovery(retrieved_at=retrieved_at, page_url=payload["page_url"], urls=urls)


def discover(
    settings: Settings | None = None,
    *,
    refresh: bool = False,
    max_age: timedelta = CACHE_MAX_AGE,
) -> Discovery:
    """Return the available quarters and their archive URLs.

    Served from the on-disk cache when it is fresh, so a re-run of the pipeline
    does not depend on the FDA site being up.
    """
    settings = settings or get_settings()
    path = cache_path(settings)

    if not refresh:
        cached = _read_cache(path)
        if cached is not None and datetime.now(tz=UTC) - cached.retrieved_at < max_age:
            log.debug("faers.discover.cache_hit", quarters=len(cached.urls))
            return cached

    response = get(EXTRACT_PAGE_URL, settings=settings, use_cache=False)
    if response.status_code != 200:
        message = f"extract page returned {response.status_code}"
        raise IngestError(message)

    urls = parse_page(response.text)
    if not urls:
        message = "extract page listed no quarterly archives; the page layout may have changed"
        raise IngestError(message)

    discovery = Discovery(retrieved_at=datetime.now(tz=UTC), page_url=EXTRACT_PAGE_URL, urls=urls)
    _write_cache(discovery, path)
    log.info(
        "faers.discover.completed",
        quarters=len(urls),
        earliest=str(min(urls)),
        latest=str(max(urls)),
    )
    return discovery


def url_for(quarter: Quarter, discovery: Discovery) -> str:
    """The archive URL for a quarter, falling back to the constructed spelling."""
    url = discovery.urls.get(quarter)
    if url is not None:
        return url
    fallback = constructed_url(quarter)
    log.warning("faers.discover.url_constructed", quarter=str(quarter), url=fallback)
    return fallback
