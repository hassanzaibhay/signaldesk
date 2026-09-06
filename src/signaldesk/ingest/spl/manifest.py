"""The record of which drug strings have had their labels fetched.

Same model and same shape as the FAERS manifest, deliberately not the same
module: `ingest.faers.manifest` is keyed on `Quarter` throughout and belongs to
that pipeline. Sharing it would mean widening its signatures for a source it
knows nothing about.

A unit here is one openFDA query, keyed by `ScopeUnit.manifest_unit`. Several
drug strings can clean to one query and are then one unit: the first to be
reached fetches, and the rest skip the fetch and attach their own drug keys to
what it stored. Keying on the query rather than the string is what makes a
cleaner change invalidate the unit by itself.

A completed unit is skipped, a failed one is retried from the start, and
retrying is safe because `store.store_labels` writes a string's labels in one
transaction.

`row_counts` is read back by the skip path, not only written. It is the only
thing that distinguishes a query that legitimately resolved to nothing from one
whose documents cannot be found, and it is where the page-cap flag lives, because
a unit that skips its fetch has no other way to learn whether the labels it is
attaching to were truncated.

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

#: Keys in ``row_counts`` that count rows. ``row_count`` is their sum, and it is
#: computed from this tuple rather than from ``sum(row_counts.values())`` so that
#: adding a non-row entry cannot silently change what ``row_count`` means.
ROW_BEARING_KEYS: tuple[str, ...] = ("documents", "sections")

#: Where non-count facts about the fetch live inside ``row_counts``.
#:
#: Nested under a key whose value is a dict, deliberately. A flag stored flat as
#: ``0``/``1`` or as a bool is absorbed by any ``sum(row_counts.values())`` as a
#: silent off-by-one, and ``bool`` is an ``int`` so a type check does not catch it
#: either. A dict makes that sum raise ``TypeError``. The explicit
#: ``ROW_BEARING_KEYS`` tuple protects the one site being edited here; this
#: encoding protects every site that is not.
FLAGS_KEY = "flags"


@dataclass(frozen=True, slots=True)
class Decision:
    """Whether a unit needs work, and why."""

    should_ingest: bool
    reason: str
    #: Documents the completed run recorded for this unit, from ``row_counts``.
    #: None unless the row exists and is completed. The skip path needs it to
    #: tell "this query resolved to nothing" from "the documents are there but
    #: could not be found", which are otherwise the same observation.
    documents_recorded: int | None = None
    #: Whether the completed fetch stopped at the page cap with more to fetch.
    #:
    #: Three states, and the third is the point. True and False are measurements
    #: made by the unit that fetched. None means the row predates the flag being
    #: recorded, so nothing is known. A unit attaching to that row must report
    #: None rather than False: P12 excludes page-capped strings by reading this
    #: out of the artifact, and a False it never measured is a corpus defect
    #: downstream rather than a cosmetic one.
    page_capped: bool | None = None


def decide(unit: str, *, force: bool = False) -> Decision:
    """Decide whether to fetch labels for one drug string."""
    if force:
        return Decision(should_ingest=True, reason="forced")
    row = IngestManifest.objects.filter(source=SOURCE, unit=unit).first()
    if row is None:
        return Decision(should_ingest=True, reason="not ingested")
    if row.status != IngestManifest.Status.COMPLETED:
        return Decision(should_ingest=True, reason=f"previous run {row.status}")
    counts = row.row_counts if isinstance(row.row_counts, dict) else {}
    recorded = counts.get("documents")
    flags = counts.get(FLAGS_KEY)
    capped = flags.get("page_cap") if isinstance(flags, dict) else None
    return Decision(
        should_ingest=False,
        reason="already completed",
        documents_recorded=int(recorded) if isinstance(recorded, int) else None,
        page_capped=bool(capped) if isinstance(capped, bool) else None,
    )


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
    page_cap: bool,
) -> None:
    """Record a successful fetch, the counts it produced, and whether it truncated.

    ``page_cap`` is recorded rather than derived. It could be inferred from
    ``documents == MAX_PAGES * PAGE_LIMIT``, and on the corpus as it stands that
    inference is even correct - exactly three rows hold 1,000 documents and
    exactly three queries hit the cap. It is correct by coincidence: parsing
    deduplicates on ``set_id``, so one label repeated across two pages would store
    999 and read as untruncated. A fetch knows whether it stopped early; a count
    only guesses.

    ``row_count`` sums ``ROW_BEARING_KEYS`` explicitly, not every value, so the
    flags entry cannot be counted as a row.
    """
    stored: dict[str, object] = {**row_counts, FLAGS_KEY: {"page_cap": page_cap}}
    rows = sum(row_counts.get(key, 0) for key in ROW_BEARING_KEYS)
    IngestManifest.objects.filter(source=SOURCE, unit=unit).update(
        status=IngestManifest.Status.COMPLETED,
        checksum=checksum,
        row_counts=stored,
        row_count=rows,
        bytes_downloaded=bytes_downloaded,
        finished_at=datetime.now(tz=UTC),
        error="",
    )
    log.info("spl.manifest.completed", unit=unit, rows=rows, page_cap=page_cap)


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
