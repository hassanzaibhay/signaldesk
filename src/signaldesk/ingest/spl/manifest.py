"""The record of which drug strings have had their labels fetched.

Same model and same shape as the FAERS manifest, deliberately not the same
module: `ingest.faers.manifest` is keyed on `Quarter` throughout and belongs to
that pipeline. Sharing it would mean widening its signatures for a source it
knows nothing about.

A unit here is one drug string, keyed by `ScopeUnit.manifest_unit`. A completed
unit is skipped, a failed one is retried from the start, and retrying is safe
because `store.store_labels` writes a string's labels in one transaction.

There is no upstream checksum. A label has no publication digest to compare
against, and openFDA revises labels continuously, so `checksum` holds the digest
of what was stored instead: it changes when the fetched content changes, which
is the signal a reviewer wants, but it cannot be known before fetching and so is
never a reason to skip.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from signaldesk.core.logging import get_logger
from signaldesk.web.signals.models import IngestManifest

log = get_logger(__name__)

SOURCE = "openfda_spl"


@dataclass(frozen=True, slots=True)
class Decision:
    """Whether a unit needs work, and why."""

    should_ingest: bool
    reason: str


def decide(unit: str, *, force: bool = False) -> Decision:
    """Decide whether to fetch labels for one drug string."""
    if force:
        return Decision(should_ingest=True, reason="forced")
    row = IngestManifest.objects.filter(source=SOURCE, unit=unit).first()
    if row is None:
        return Decision(should_ingest=True, reason="not ingested")
    if row.status != IngestManifest.Status.COMPLETED:
        return Decision(should_ingest=True, reason=f"previous run {row.status}")
    return Decision(should_ingest=False, reason="already completed")


def start(unit: str) -> IngestManifest:
    """Mark a unit as in progress, clearing any previous outcome."""
    row, _ = IngestManifest.objects.update_or_create(
        source=SOURCE,
        unit=unit,
        defaults={
            "status": IngestManifest.Status.RUNNING,
            "error": "",
            "finished_at": None,
            "started_at": datetime.now(tz=UTC),
        },
    )
    return row


def complete(
    unit: str,
    *,
    checksum: str,
    row_counts: dict[str, int],
    bytes_downloaded: int,
) -> None:
    """Record a successful fetch and the counts it produced."""
    IngestManifest.objects.filter(source=SOURCE, unit=unit).update(
        status=IngestManifest.Status.COMPLETED,
        checksum=checksum,
        row_counts=row_counts,
        row_count=sum(row_counts.values()),
        bytes_downloaded=bytes_downloaded,
        finished_at=datetime.now(tz=UTC),
        error="",
    )
    log.info("spl.manifest.completed", unit=unit, rows=sum(row_counts.values()))


def fail(unit: str, error: str) -> None:
    """Record a failure so the next run retries the unit."""
    IngestManifest.objects.filter(source=SOURCE, unit=unit).update(
        status=IngestManifest.Status.FAILED,
        error=error[:2000],
        finished_at=datetime.now(tz=UTC),
    )
    log.error("spl.manifest.failed", unit=unit, error=error[:200])


def completed_units() -> set[str]:
    """Every unit successfully ingested."""
    return set(
        IngestManifest.objects.filter(
            source=SOURCE, status=IngestManifest.Status.COMPLETED
        ).values_list("unit", flat=True)
    )
