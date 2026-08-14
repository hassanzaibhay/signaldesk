"""Fetch and unpack one quarterly archive at a time.

Two constraints shape this module.

Disk: a single recent quarter is a few hundred megabytes compressed and roughly a
gigabyte extracted, and there are 55 of them. Keeping two on disk at once is how
a laptop runs out of space mid-run, so every caller uses `staged_quarter`, which
deletes both the archive and the extracted tree on the way out, including when
the body raised.

Throughput: the source serves a single connection at roughly 40 KB/s, measured
from two machines, which puts the full corpus somewhere past thirty hours. It
does advertise `Accept-Ranges: bytes`, and splitting a file across a few
connections scales close to linearly, so a large archive is fetched in segments.
The segment count is deliberately small: this is a public service, and the point
is to stop wasting a day per corpus refresh, not to extract everything the server
will give.
"""

from __future__ import annotations

import hashlib
import shutil
import zipfile
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from signaldesk.core.config import Settings, get_settings
from signaldesk.core.errors import IngestError
from signaldesk.core.http import build_client
from signaldesk.core.logging import get_logger
from signaldesk.ingest.faers.quarter import Quarter

log = get_logger(__name__)

CHUNK_BYTES = 1 << 20
DOWNLOAD_ATTEMPTS = 3

#: A quarterly archive is tens of megabytes at minimum. Anything smaller is an
#: error page that arrived with a 200, which has happened to this source before.
MIN_PLAUSIBLE_BYTES = 1 << 20

#: Connections used for a segmented fetch. Four is enough to lift throughput from
#: unusable to tolerable and stays a reasonable neighbour on a public service.
SEGMENTS = 4

#: Below this, one connection is simpler and the setup cost is not worth paying.
SEGMENT_THRESHOLD_BYTES = 8 << 20

#: This host is slow to answer, not only slow to transfer: a one-byte ranged GET
#: has taken over 30 seconds to produce headers. The shared client's defaults are
#: tuned for APIs and time out well before this source replies.
BULK_TIMEOUT = httpx.Timeout(connect=30.0, read=300.0, write=60.0, pool=30.0)


#: A segment is re-requested from where it stopped rather than from the start,
#: because this host closes connections mid-body regularly enough that whole-file
#: restarts would rarely finish.
SEGMENT_ATTEMPTS = 8


def _bulk_client(settings: Settings) -> httpx.Client:
    """The shared client, tuned for this host.

    `Accept-Encoding: identity` is not optional here. With the default
    negotiation header the server stalls and the request times out; asking for no
    encoding returns the bytes immediately. Compressing a zip was never going to
    help anyone.
    """
    client = build_client(settings, timeout=BULK_TIMEOUT)
    client.headers["Accept-Encoding"] = "identity"
    return client


@dataclass(frozen=True, slots=True)
class Archive:
    """A downloaded archive and what it hashed to."""

    quarter: Quarter
    path: Path
    sha256: str
    size_bytes: int


def raw_dir(settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    return settings.data_dir / "raw" / "faers"


def remote_size(url: str, settings: Settings | None = None) -> int | None:
    """Total size in bytes, or None when the server will not say.

    A HEAD against this host returns `Content-Length: 0`, so the size comes from
    the `Content-Range` of a one-byte ranged GET instead.
    """
    settings = settings or get_settings()
    with _bulk_client(settings) as client:
        response = client.get(url, headers={"Range": "bytes=0-0"})
    if response.status_code != httpx.codes.PARTIAL_CONTENT:
        return None
    content_range = response.headers.get("Content-Range", "")
    _, _, total = content_range.partition("/")
    if not total.isdigit():
        return None
    return int(total)


def _fetch_segment(url: str, start: int, end: int, path: Path, settings: Settings) -> int:
    """Fetch one byte range into its own part file, resuming where it stops.

    The server truncates responses without warning, so each attempt asks only for
    the bytes still missing and appends them. Progress is kept across attempts;
    an attempt that returns nothing counts against the budget so a permanently
    stuck segment still fails rather than looping.
    """
    expected = end - start + 1
    written = 0
    path.unlink(missing_ok=True)

    for attempt in range(1, SEGMENT_ATTEMPTS + 1):
        before = written
        try:
            with _bulk_client(settings) as client:
                headers = {"Range": f"bytes={start + written}-{end}"}
                with client.stream("GET", url, headers=headers) as response:
                    if response.status_code != httpx.codes.PARTIAL_CONTENT:
                        message = (
                            f"segment {start}-{end} returned {response.status_code}, expected 206"
                        )
                        raise IngestError(message)
                    with path.open("ab") as handle:
                        for chunk in response.iter_bytes(CHUNK_BYTES):
                            handle.write(chunk)
                            written += len(chunk)
        except httpx.TransportError as exc:
            log.warning(
                "faers.download.segment_interrupted",
                start=start,
                end=end,
                attempt=attempt,
                have=written,
                want=expected,
                error=type(exc).__name__,
            )
            if attempt == SEGMENT_ATTEMPTS:
                raise
            continue

        if written >= expected:
            break
        if written == before:
            log.warning("faers.download.segment_stalled", start=start, end=end, attempt=attempt)
        if attempt == SEGMENT_ATTEMPTS:
            message = f"segment {start}-{end} stopped at {written} of {expected} bytes"
            raise IngestError(message)

    if written != expected:
        message = f"segment {start}-{end} returned {written} bytes, expected {expected}"
        raise IngestError(message)
    return written


def _download_segmented(
    url: str, path: Path, total: int, settings: Settings, *, segments: int
) -> None:
    """Fetch the file as N byte ranges in parallel, then join them in order."""
    span = total // segments
    bounds = [
        (index * span, (total - 1) if index == segments - 1 else ((index + 1) * span - 1))
        for index in range(segments)
    ]
    parts = [path.with_suffix(f".part{index}") for index in range(segments)]

    try:
        with ThreadPoolExecutor(max_workers=segments) as pool:
            futures = [
                pool.submit(_fetch_segment, url, start, end, part, settings)
                for (start, end), part in zip(bounds, parts, strict=True)
            ]
            for future in futures:
                future.result()

        with path.open("wb") as target:
            for part in parts:
                with part.open("rb") as source:
                    shutil.copyfileobj(source, target, CHUNK_BYTES)
    finally:
        for part in parts:
            part.unlink(missing_ok=True)


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


@retry(
    retry=retry_if_exception_type((httpx.TransportError, IngestError)),
    stop=stop_after_attempt(DOWNLOAD_ATTEMPTS),
    wait=wait_exponential_jitter(initial=2.0, max=30.0),
    reraise=True,
)
def download_archive(
    quarter: Quarter,
    url: str,
    settings: Settings | None = None,
    *,
    destination: Path | None = None,
    segments: int = SEGMENTS,
) -> Archive:
    """Fetch one quarterly archive to disk and hash it.

    Large archives are fetched as parallel byte ranges; small ones use a single
    connection. A failure retries the whole archive rather than resuming, because
    a quarter is small enough that restarting beats tracking partial state.
    """
    settings = settings or get_settings()
    target_dir = destination or raw_dir(settings)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{quarter.label}.zip"

    total = remote_size(url, settings)
    if total is not None and total >= SEGMENT_THRESHOLD_BYTES:
        log.info(
            "faers.download.started",
            quarter=quarter.label,
            bytes=total,
            segments=segments,
        )
        _download_segmented(url, path, total, settings, segments=segments)
    else:
        log.info("faers.download.started", quarter=quarter.label, bytes=total, segments=1)
        with _bulk_client(settings) as client, client.stream("GET", url) as response:
            if response.status_code != httpx.codes.OK:
                message = f"{quarter}: archive request returned {response.status_code} for {url}"
                raise IngestError(message)
            with path.open("wb") as handle:
                for chunk in response.iter_bytes(CHUNK_BYTES):
                    handle.write(chunk)

    checksum, written = _hash_file(path)

    if written < MIN_PLAUSIBLE_BYTES:
        path.unlink(missing_ok=True)
        message = f"{quarter}: archive was only {written} bytes, which cannot be a quarter"
        raise IngestError(message)
    if total is not None and written != total:
        path.unlink(missing_ok=True)
        message = f"{quarter}: truncated download, expected {total} bytes, got {written}"
        raise IngestError(message)
    if not zipfile.is_zipfile(path):
        path.unlink(missing_ok=True)
        message = f"{quarter}: downloaded file is not a zip archive"
        raise IngestError(message)

    archive = Archive(quarter=quarter, path=path, sha256=checksum, size_bytes=written)
    log.info(
        "faers.download.completed",
        quarter=quarter.label,
        bytes=written,
        sha256=archive.sha256[:12],
    )
    return archive


def extract_archive(archive: Archive, *, destination: Path | None = None) -> Path:
    """Unpack the archive into a directory named for its quarter.

    Entries are checked against the destination before extraction: a zip is an
    untrusted input, and an entry named "../.." would otherwise write outside it.
    """
    target = destination or archive.path.with_suffix("")
    target.mkdir(parents=True, exist_ok=True)
    resolved_target = target.resolve()

    with zipfile.ZipFile(archive.path) as bundle:
        for member in bundle.namelist():
            candidate = (resolved_target / member).resolve()
            if not candidate.is_relative_to(resolved_target):
                message = f"{archive.quarter}: archive entry escapes the target directory: {member}"
                raise IngestError(message)
        bundle.extractall(resolved_target)

    log.info("faers.extract.completed", quarter=archive.quarter.label, path=str(target))
    return target


def remove_quarter(quarter: Quarter, settings: Settings | None = None) -> int:
    """Delete a quarter's archive and extracted tree. Returns bytes reclaimed."""
    settings = settings or get_settings()
    base = raw_dir(settings)
    reclaimed = 0

    archive_path = base / f"{quarter.label}.zip"
    if archive_path.is_file():
        reclaimed += archive_path.stat().st_size
        archive_path.unlink()

    extracted = base / quarter.label
    if extracted.is_dir():
        reclaimed += sum(item.stat().st_size for item in extracted.rglob("*") if item.is_file())
        shutil.rmtree(extracted)

    if reclaimed:
        log.info("faers.raw.removed", quarter=quarter.label, bytes=reclaimed)
    return reclaimed


@contextmanager
def staged_quarter(
    quarter: Quarter,
    url: str,
    settings: Settings | None = None,
    *,
    keep_raw: bool = False,
) -> Iterator[tuple[Path, Archive]]:
    """Download and extract a quarter, then delete it however the body exits.

    Yields the extracted directory together with the `Archive` record, because
    the checksum and byte count have to reach the manifest and the files
    themselves are gone by the time this returns.

    `keep_raw` leaves the files behind for debugging a single quarter. It is not
    a mode to run a range in: 55 quarters retained is well over a terabyte.
    """
    settings = settings or get_settings()
    archive = download_archive(quarter, url, settings)
    try:
        yield extract_archive(archive), archive
    finally:
        if keep_raw:
            log.warning(
                "faers.raw.retained",
                quarter=quarter.label,
                path=str(raw_dir(settings)),
                reason="keep_raw requested; delete these files before ingesting a range",
            )
        else:
            remove_quarter(quarter, settings)
