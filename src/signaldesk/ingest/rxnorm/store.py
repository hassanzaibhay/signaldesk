"""Writing string matches to Postgres, idempotently.

Normalization is re-run: after a cache warm-up, after an override is added, after
the cleaning rules change. So writing has to converge rather than accumulate, and
the shape that guarantees it is an upsert keyed on the string plus a full replace
of that string's ingredient rows. Appending would leave an ingredient behind
after a mapping was corrected, and the corrected case would then carry both the
old and the new concept with nothing to say which is current.

Kept separate from `match` so the matcher stays free of the ORM and can be tested
without a database.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import islice

from django.db import transaction

from signaldesk.core.logging import get_logger
from signaldesk.ingest.rxnorm.match import StringMatch
from signaldesk.web.signals.models import DrugConcept, DrugStringIngredient, DrugStringMatch

log = get_logger(__name__)

#: Rows per transaction. Large enough to amortise the round trips, small enough
#: that a failure loses a batch rather than a pass.
BATCH_ROWS = 1_000


@dataclass(frozen=True, slots=True)
class StoreCounts:
    """What one write pass changed."""

    strings: int
    concepts: int
    associations: int


def _batched(matches: Iterable[StringMatch], size: int) -> Iterator[list[StringMatch]]:
    iterator = iter(matches)
    while batch := list(islice(iterator, size)):
        yield batch


def store_matches(matches: Iterable[StringMatch], *, batch_size: int = BATCH_ROWS) -> StoreCounts:
    """Upsert string matches and replace their ingredient associations."""
    strings = concepts = associations = 0

    for batch in _batched(matches, batch_size):
        with transaction.atomic():
            concepts += _upsert_concepts(batch)
            strings += _upsert_matches(batch)
            associations += _replace_associations(batch)

    log.info("rxnorm.store.written", strings=strings, concepts=concepts, associations=associations)
    return StoreCounts(strings=strings, concepts=concepts, associations=associations)


def _upsert_concepts(batch: list[StringMatch]) -> int:
    """Ingredient concepts, deduplicated within the batch.

    Names are refreshed on conflict: RxNorm revises them, and a stale name beside
    a current identifier is the kind of mismatch that gets read as a bug in the
    mapping rather than as a vocabulary update.
    """
    concepts = {
        int(concept.rxcui): DrugConcept(
            ingredient_rxcui=int(concept.rxcui), name=concept.name, tty=concept.tty
        )
        for match in batch
        for concept in match.ingredients
    }
    if not concepts:
        return 0
    DrugConcept.objects.bulk_create(
        list(concepts.values()),
        update_conflicts=True,
        update_fields=["name", "tty"],
        unique_fields=["ingredient_rxcui"],
    )
    return len(concepts)


def _upsert_matches(batch: list[StringMatch]) -> int:
    rows = [
        DrugStringMatch(
            source_field=match.source_field,
            folded_string=match.folded,
            cleaned_string=match.cleaned,
            rxcui=int(match.rxcui) if match.rxcui else None,
            match_method=match.method,
            match_score=match.score,
            candidate_rank=match.rank,
            components_total=match.components_total,
            components_matched=match.components_matched,
        )
        for match in batch
    ]
    DrugStringMatch.objects.bulk_create(
        rows,
        update_conflicts=True,
        update_fields=[
            "cleaned_string",
            "rxcui",
            "match_method",
            "match_score",
            "candidate_rank",
            "components_total",
            "components_matched",
            "retrieved_at",
        ],
        unique_fields=["source_field", "folded_string"],
    )
    return len(rows)


def _replace_associations(batch: list[StringMatch]) -> int:
    """Rewrite each string's ingredient rows, rather than adding to them."""
    keys = {(match.source_field, match.folded) for match in batch}
    identifiers = {
        (row.source_field, row.folded_string): row.pk
        for row in DrugStringMatch.objects.filter(
            source_field__in={key[0] for key in keys},
            folded_string__in={key[1] for key in keys},
        ).only("id", "source_field", "folded_string")
    }

    match_ids = [identifiers[key] for key in keys if key in identifiers]
    DrugStringIngredient.objects.filter(match_id__in=match_ids).delete()

    rows = [
        DrugStringIngredient(
            match_id=identifiers[(match.source_field, match.folded)],
            ingredient_id=int(concept.rxcui),
            ordinal=ordinal,
        )
        for match in batch
        if (match.source_field, match.folded) in identifiers
        for ordinal, concept in enumerate(match.ingredients)
    ]
    DrugStringIngredient.objects.bulk_create(rows)
    return len(rows)
