"""Writing labels to Postgres, idempotently.

The pipeline is re-run: after the scope widens, after an override is added,
after the cleaner changes. So writing converges rather than accumulates. Two
rules, both the same shape as the RxNorm store:

* a document is upserted on ``set_id``, never appended;
* its sections are replaced wholesale, never merged. A revised label with a
  shorter adverse-reactions section would otherwise keep the paragraphs the
  revision removed, and the document would carry text no published label says.

Kept apart from `parse` so the parser stays free of the ORM and testable without
a database.
"""

from __future__ import annotations

from dataclasses import dataclass

from django.db import transaction

from signaldesk.core.logging import get_logger
from signaldesk.ingest.spl.parse import LabelRecord
from signaldesk.web.documents.models import LabelDocument, LabelDrugKey, LabelSection

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class StoreCounts:
    """What one write pass changed."""

    documents: int
    sections: int
    drug_keys: int


def store_labels(
    records: list[LabelRecord],
    *,
    folded_string: str,
    query: str,
    route: str,
    ingredient_rxcui: int | None = None,
) -> StoreCounts:
    """Upsert every label found for one drug string, and key the string to them.

    One transaction for the whole string, so a failure part way through leaves
    the string looking un-ingested rather than half-ingested. The manifest is
    what decides whether it is retried, and a half-written string that the
    manifest called complete is the failure mode this avoids.
    """
    documents = sections = drug_keys = 0

    with transaction.atomic():
        for record in records:
            document, _ = LabelDocument.objects.update_or_create(
                set_id=record.set_id,
                defaults={
                    "spl_id": record.spl_id,
                    "version": record.version,
                    "effective_time": record.effective_time,
                    "brand_names": record.brand_names,
                    "generic_names": record.generic_names,
                    "substance_names": record.substance_names,
                    "rxcuis": record.rxcuis,
                },
            )
            documents += 1

            LabelSection.objects.filter(document=document).delete()
            if record.sections:
                LabelSection.objects.bulk_create(
                    [
                        LabelSection(
                            document=document,
                            section_code=section.section_code,
                            ordinal=section.ordinal,
                            text=section.text,
                        )
                        for section in record.sections
                    ]
                )
                sections += len(record.sections)

            LabelDrugKey.objects.update_or_create(
                folded_string=folded_string,
                document=document,
                defaults={
                    "query": query,
                    "route": route,
                    "ingredient_rxcui": ingredient_rxcui,
                },
            )
            drug_keys += 1

    log.info(
        "spl.store.written",
        folded_string=folded_string,
        documents=documents,
        sections=sections,
        drug_keys=drug_keys,
    )
    return StoreCounts(documents=documents, sections=sections, drug_keys=drug_keys)


def attach_drug_keys(
    *,
    folded_string: str,
    query: str,
    route: str,
    ingredient_rxcui: int | None = None,
) -> int:
    """Key a string to the documents another string's fetch already stored.

    Several drug strings can clean to one query, and the manifest is keyed on
    the query, so only the first is fetched. The rest still need their own
    reach-through: the FAERS string is the join key back to the signal table,
    and a lookup on the exact string has to resolve.

    The documents are found by ``query``, which is the manifest identity - every
    string in a group asked openFDA the same thing and reached the same labels.

    Writes ``LabelDrugKey`` and nothing else. No ``LabelDocument``, no
    ``LabelSection``: those belong to the unit that fetched them, and re-storing
    them here would put the run's distinct document and section counts wrong
    again in a new way.

    Returns the number of distinct documents this string now reaches, which is
    also the number of rows written.

    ``order_by()`` is load-bearing and must not be dropped. ``Meta.ordering`` on
    ``LabelDrugKey`` is ``["folded_string", "document"]``, and Django appends the
    ordering columns to a ``values_list(...).distinct()`` so the database can sort
    by them. The DISTINCT is then over ``(document_id, folded_string, set_id)``
    rather than over ``document_id``, and the list comes back with one entry per
    (document, string) pair. On the real corpus that is a clean multiple: the
    ``GABAPENTIN`` query returned 828 entries for 414 documents once two strings
    were keyed to it. Stripping the ordering makes the DISTINCT mean what it says.

    The rows written were always right, because ``update_or_create`` is keyed on
    (folded_string, document) and a repeated id is a no-op. Only the count was
    wrong, which is worse in one specific way: it is the number the run artifact
    reports, so it was inflated in the artifact while the database was correct.
    """
    document_ids = list(
        LabelDrugKey.objects.filter(query=query)
        .order_by()
        .values_list("document_id", flat=True)
        .distinct()
    )
    written = 0
    with transaction.atomic():
        for document_id in document_ids:
            LabelDrugKey.objects.update_or_create(
                folded_string=folded_string,
                document_id=document_id,
                defaults={
                    "query": query,
                    "route": route,
                    "ingredient_rxcui": ingredient_rxcui,
                },
            )
            written += 1
    log.info(
        "spl.store.keys_attached",
        folded_string=folded_string,
        query=query,
        drug_keys=written,
    )
    return written


def sections_for(folded_string: str, codes: tuple[str, ...] | None = None) -> list[LabelSection]:
    """Every stored section reachable from one drug string.

    The lookup the retrieval layer makes. Indexed on ``folded_string``, so this
    is one index scan and a join rather than a query against openFDA.
    """
    query = LabelSection.objects.filter(document__drug_keys__folded_string=folded_string)
    if codes:
        query = query.filter(section_code__in=codes)
    return list(query.select_related("document").distinct())
