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


def sections_for(folded_string: str, codes: tuple[str, ...] | None = None) -> list[LabelSection]:
    """Every stored section reachable from one drug string.

    The lookup the retrieval layer makes. Indexed on ``folded_string``, so this
    is one index scan and a join rather than a query against openFDA.
    """
    query = LabelSection.objects.filter(document__drug_keys__folded_string=folded_string)
    if codes:
        query = query.filter(section_code__in=codes)
    return list(query.select_related("document").distinct())
