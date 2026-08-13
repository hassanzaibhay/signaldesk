"""Fetching and unpacking archives, including the ways this host misbehaves.

The interesting cases here are all failure cases: a truncated body, a stalled
segment, an archive that is not an archive, and a zip entry trying to escape its
directory. Those are what the retry and validation logic exists for.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import httpx
import pytest
import respx

from signaldesk.core.config import Settings
from signaldesk.core.errors import IngestError
from signaldesk.ingest.faers.download import (
    Archive,
    download_archive,
    extract_archive,
    raw_dir,
    remote_size,
    remove_quarter,
    staged_quarter,
)
from signaldesk.ingest.faers.quarter import Quarter

pytestmark = pytest.mark.unit

URL = "https://fis.fda.test/faers_ascii_2013q1.zip"
QUARTER = Quarter.parse("2013Q1")


def _zip_bytes(names: tuple[str, ...] = ("ascii/DEMO13Q1.txt",), size: int = 2 << 20) -> bytes:
    """A zip large enough to pass the implausibly-small check.

    Stored rather than deflated, and filled with incompressible bytes, so the
    archive really is the size asked for: the downloader rejects anything under
    a megabyte as an error page that arrived with a 200.
    """
    import io
    import os

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as bundle:
        for name in names:
            bundle.writestr(name, os.urandom(size))
    return buffer.getvalue()


def _server(body: bytes, *, ranges: bool = False):
    """Stand in for the archive host, answering however many calls arrive."""

    def respond(request: httpx.Request) -> httpx.Response:
        header = request.headers.get("Range")
        if header is None:
            return httpx.Response(200, content=body)
        start, _, end = header.removeprefix("bytes=").partition("-")
        if not ranges:
            return httpx.Response(200, content=body)
        first = int(start)
        last = int(end) if end else len(body) - 1
        return httpx.Response(
            206,
            content=body[first : last + 1],
            headers={"Content-Range": f"bytes {first}-{last}/{len(body)}"},
        )

    return respond


@respx.mock
def test_remote_size_reads_the_content_range(settings: Settings) -> None:
    respx.get(URL).mock(
        return_value=httpx.Response(206, headers={"Content-Range": "bytes 0-0/25915184"})
    )
    assert remote_size(URL, settings) == 25_915_184


@respx.mock
def test_remote_size_is_unknown_when_ranges_are_unsupported(settings: Settings) -> None:
    respx.get(URL).mock(return_value=httpx.Response(200))
    assert remote_size(URL, settings) is None


@respx.mock
def test_remote_size_is_unknown_when_the_header_is_malformed(settings: Settings) -> None:
    respx.get(URL).mock(
        return_value=httpx.Response(206, headers={"Content-Range": "bytes 0-0/unknown"})
    )
    assert remote_size(URL, settings) is None


@respx.mock
def test_small_archive_uses_a_single_request(settings: Settings) -> None:
    payload = _zip_bytes()
    respx.get(URL).mock(side_effect=_server(payload))
    archive = download_archive(QUARTER, URL, settings)

    assert archive.size_bytes == len(payload)
    assert archive.path.read_bytes() == payload


@respx.mock
def test_identity_encoding_is_requested(settings: Settings) -> None:
    payload = _zip_bytes()
    route = respx.get(URL).mock(side_effect=_server(payload))
    download_archive(QUARTER, URL, settings)
    # Without this the server accepts the connection and never responds.
    assert route.calls.last.request.headers["Accept-Encoding"] == "identity"


@respx.mock
def test_a_body_that_is_not_a_zip_is_rejected(settings: Settings) -> None:
    # Large enough to look plausible, so it is the zip check that rejects it.
    respx.get(URL).mock(side_effect=_server(b"<html>error</html>" * 100_000))
    with pytest.raises(IngestError, match="not a zip archive"):
        download_archive(QUARTER, URL, settings)


@respx.mock
def test_a_tiny_body_is_rejected(settings: Settings) -> None:
    respx.get(URL).mock(side_effect=_server(b"nope"))
    with pytest.raises(IngestError, match="cannot be a quarter"):
        download_archive(QUARTER, URL, settings)


def test_extract_writes_the_archive_contents(settings: Settings, tmp_path: Path) -> None:
    path = tmp_path / "2013Q1.zip"
    path.write_bytes(_zip_bytes())
    archive = Archive(quarter=QUARTER, path=path, sha256="abc", size_bytes=path.stat().st_size)

    root = extract_archive(archive)
    assert (root / "ascii" / "DEMO13Q1.txt").is_file()


def test_an_entry_escaping_the_target_is_refused(settings: Settings, tmp_path: Path) -> None:
    path = tmp_path / "2013Q1.zip"
    path.write_bytes(_zip_bytes(names=("../escaped.txt",)))
    archive = Archive(quarter=QUARTER, path=path, sha256="abc", size_bytes=path.stat().st_size)

    with pytest.raises(IngestError, match="escapes the target directory"):
        extract_archive(archive)


def test_remove_quarter_reclaims_both_archive_and_tree(settings: Settings) -> None:
    base = raw_dir(settings)
    base.mkdir(parents=True, exist_ok=True)
    (base / "2013Q1.zip").write_bytes(b"z" * 100)
    extracted = base / "2013Q1" / "ascii"
    extracted.mkdir(parents=True)
    (extracted / "DEMO13Q1.txt").write_bytes(b"d" * 50)

    reclaimed = remove_quarter(QUARTER, settings)

    assert reclaimed == 150
    assert not (base / "2013Q1.zip").exists()
    assert not (base / "2013Q1").exists()


def test_remove_quarter_on_nothing_is_zero(settings: Settings) -> None:
    assert remove_quarter(QUARTER, settings) == 0


@respx.mock
def test_staged_quarter_deletes_raw_files_even_when_the_body_raises(settings: Settings) -> None:
    respx.get(URL).mock(side_effect=_server(_zip_bytes()))

    with pytest.raises(RuntimeError, match="boom"), staged_quarter(QUARTER, URL, settings):
        raise RuntimeError("boom")

    assert not (raw_dir(settings) / "2013Q1.zip").exists()
    assert not (raw_dir(settings) / "2013Q1").exists()


@respx.mock
def test_keep_raw_retains_the_files(settings: Settings) -> None:
    respx.get(URL).mock(side_effect=_server(_zip_bytes()))

    with staged_quarter(QUARTER, URL, settings, keep_raw=True) as (root, archive):
        assert root.is_dir()

    assert archive.path.exists()


@respx.mock
def test_a_large_archive_is_fetched_in_segments(settings: Settings) -> None:
    payload = _zip_bytes(size=12 << 20)
    route = respx.get(URL).mock(side_effect=_server(payload, ranges=True))

    archive = download_archive(QUARTER, URL, settings, segments=4)

    assert archive.path.read_bytes() == payload
    # One sizing request plus one per segment.
    assert route.call_count == 5
