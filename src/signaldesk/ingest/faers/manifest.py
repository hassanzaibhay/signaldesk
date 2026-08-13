"""The record of what has been ingested, which makes re-running safe.

These pipelines get re-run constantly: a quarter fails halfway, a parser bug is
fixed, the corpus is extended. The manifest is what makes that cheap. A unit
that completed and whose upstream bytes still hash the same is skipped. A unit
that failed is retried from the start, which is safe because a quarter is
written as a whole. A unit whose upstream checksum changed is re-ingested, and
the change is logged rather than passed over, because the source silently
revising a published quarter is something a reviewer needs to know happened.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from signaldesk.core.logging import get_logger
from signaldesk.ingest.faers.quarter import Quarter
from signaldesk.web.signals.models import IngestManifest

log = get_logger(__name__)

SOURCE = "faers"


@dataclass(frozen=True, slots=True)
class Decision:
    """Whether a unit needs work, and why."""

    should_ingest: bool
    reason: str


def decide(quarter: Quarter, checksum: str | None, *, force: bool = False) -> Decision:
    """Decide whether to ingest a quarter.

    `checksum` is the upstream hash when it is already known. It is optional
    because discovering it costs a download, and a completed quarter should not
    be downloaded merely to confirm it can be skipped.
    """
    row = IngestManifest.objects.filter(source=SOURCE, unit=quarter.label).first()
    if force:
        return Decision(should_ingest=True, reason="forced")
    if row is None:
        return Decision(should_ingest=True, reason="not ingested")
    if row.status != IngestManifest.Status.COMPLETED:
        return Decision(should_ingest=True, reason=f"previous run {row.status}")
    if checksum is not None and row.checksum and row.checksum != checksum:
        log.warning(
            "faers.manifest.checksum_changed",
            quarter=quarter.label,
            previous=row.checksum[:12],
            current=checksum[:12],
        )
        return Decision(should_ingest=True, reason="upstream checksum changed")
    return Decision(should_ingest=False, reason="already completed")


def start(quarter: Quarter) -> IngestManifest:
    """Mark a unit as in progress, clearing any previous outcome."""
    row, _ = IngestManifest.objects.update_or_create(
        source=SOURCE,
        unit=quarter.label,
        defaults={
            "status": IngestManifest.Status.RUNNING,
            "error": "",
            "finished_at": None,
            "started_at": datetime.now(tz=UTC),
        },
    )
    return row


def complete(
    quarter: Quarter,
    *,
    checksum: str,
    row_counts: dict[str, int],
    bytes_downloaded: int,
    had_deleted_file: bool,
) -> None:
    """Record a successful ingest and the counts it produced."""
    IngestManifest.objects.filter(source=SOURCE, unit=quarter.label).update(
        status=IngestManifest.Status.COMPLETED,
        checksum=checksum,
        row_counts=row_counts,
        row_count=sum(row_counts.values()),
        bytes_downloaded=bytes_downloaded,
        had_deleted_file=had_deleted_file,
        finished_at=datetime.now(tz=UTC),
        error="",
    )
    log.info("faers.manifest.completed", quarter=quarter.label, rows=sum(row_counts.values()))


def fail(quarter: Quarter, error: str) -> None:
    """Record a failed ingest so the next run retries it."""
    IngestManifest.objects.filter(source=SOURCE, unit=quarter.label).update(
        status=IngestManifest.Status.FAILED,
        error=error[:2000],
        finished_at=datetime.now(tz=UTC),
    )
    log.error("faers.manifest.failed", quarter=quarter.label, error=error[:200])


def completed_quarters() -> list[Quarter]:
    """Every quarter successfully ingested, ascending."""
    labels = IngestManifest.objects.filter(
        source=SOURCE, status=IngestManifest.Status.COMPLETED
    ).values_list("unit", flat=True)
    return sorted(Quarter.parse(label) for label in labels)
