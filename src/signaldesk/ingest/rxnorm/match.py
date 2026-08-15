"""Resolving one drug string to ingredient concepts, in priority order.

## The order, and why it is an order rather than a score

    1. curated override
    2. exact term match
    3. approximate term match
    4. unmatched

Each rung is a different kind of claim, and the probes show they are not
comparable on one scale. `UNSPECIFIED INGREDIENT` returns nothing from the exact
endpoint and `caviar preparation` from the approximate one, at a score higher
than a correct match on a German ingredient name. So the exact rung is tried
first for every string, not as an optimisation but because it refuses what it
does not know, and every string it answers is a string kept out of the rung where
fabricated matches live.

Note what this module does *not* do: it does not rank `prod_ai` above `drugname`.
That is a property of a drug row, not of a string, and the mapping is stored per
string (there are 84,589,952 rows and 656,589 distinct strings). Each string is
resolved once for the field it appeared in, and the row-level priority - prod_ai
exact, drugname exact, prod_ai approximate, drugname approximate - is applied by
the join that reads these rows back. The rung breakdown in the coverage report is
`source_field` crossed with `match_method`, which is the same information counted
where it belongs.

## Combinations

A string naming several ingredients produces several associations, never one
arbitrary winner. When only some components resolve, the ones that did are kept
and both counts are recorded, because a partially resolved combination is a
silently shortened ingredient set otherwise.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from signaldesk.core.config import Settings, get_settings
from signaldesk.core.errors import NormalizationError
from signaldesk.core.logging import get_logger
from signaldesk.ingest.rxnorm import client
from signaldesk.ingest.rxnorm.normalize import clean

log = get_logger(__name__)

DISPOSITION_INGREDIENT = "ingredient"
DISPOSITION_NOT_A_DRUG = "not_a_drug"

METHOD_OVERRIDE = "override"
METHOD_OVERRIDE_NOT_A_DRUG = "override_not_a_drug"
METHOD_EXACT = "exact"
METHOD_APPROXIMATE = "approximate"
METHOD_UNMATCHED = "unmatched"

OVERRIDES_PATH = Path("data/overrides/drug_overrides.csv")
OVERRIDE_COLUMNS = ("raw_string", "disposition", "ingredient_rxcui", "ingredient_name", "note")


@dataclass(frozen=True, slots=True)
class Override:
    """One curated decision about one string.

    `not_a_drug` exists because no threshold can catch a string that is not a
    drug and still resolves to a real ingredient. It is a positive statement
    that the string names nothing, and it is the only way the file can say so.
    """

    raw_string: str
    disposition: str
    ingredient_rxcui: int | None
    ingredient_name: str
    note: str


@dataclass(frozen=True, slots=True)
class StringMatch:
    """What one distinct source string resolved to, and by which rung."""

    source_field: str
    folded: str
    cleaned: str
    method: str
    rxcui: str | None = None
    ingredients: tuple[client.Concept, ...] = ()
    score: float | None = None
    rank: int | None = None
    components_total: int = 1
    components_matched: int = 0
    requests: int = 0
    cache_hits: int = 0
    steps: tuple[str, ...] = field(default_factory=tuple)

    @property
    def matched(self) -> bool:
        return self.method in {METHOD_OVERRIDE, METHOD_EXACT, METHOD_APPROXIMATE}

    @property
    def is_partial(self) -> bool:
        """Some components of a combination resolved and others did not."""
        return 0 < self.components_matched < self.components_total


def load_overrides(path: Path = OVERRIDES_PATH) -> dict[str, Override]:
    """Read the curated table, keyed on the folded raw string.

    Curated by hand, so it is validated rather than trusted: a row that cannot
    mean what it says raises instead of being skipped, since a silently dropped
    override is a mapping nobody notices is missing.
    """
    if not path.is_file():
        return {}

    overrides: dict[str, Override] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(OVERRIDE_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            message = f"{path} is missing columns: {', '.join(sorted(missing))}"
            raise NormalizationError(message)

        for number, row in enumerate(reader, start=2):
            raw = (row["raw_string"] or "").strip().upper()
            disposition = (row["disposition"] or "").strip()
            rxcui = (row["ingredient_rxcui"] or "").strip()
            note = (row["note"] or "").strip()
            if not raw:
                message = f"{path} line {number}: empty raw_string"
                raise NormalizationError(message)
            if disposition == DISPOSITION_INGREDIENT:
                if not rxcui.isdigit():
                    message = f"{path} line {number}: disposition 'ingredient' needs an rxcui"
                    raise NormalizationError(message)
            elif disposition == DISPOSITION_NOT_A_DRUG:
                if rxcui:
                    message = f"{path} line {number}: 'not_a_drug' cannot carry an rxcui"
                    raise NormalizationError(message)
                if not note:
                    message = f"{path} line {number}: 'not_a_drug' needs a note explaining why"
                    raise NormalizationError(message)
            else:
                message = (
                    f"{path} line {number}: disposition must be "
                    f"'{DISPOSITION_INGREDIENT}' or '{DISPOSITION_NOT_A_DRUG}', got {disposition!r}"
                )
                raise NormalizationError(message)

            overrides[raw] = Override(
                raw_string=raw,
                disposition=disposition,
                ingredient_rxcui=int(rxcui) if rxcui else None,
                ingredient_name=(row["ingredient_name"] or "").strip(),
                note=note,
            )

    log.info("rxnorm.overrides.loaded", path=str(path), entries=len(overrides))
    return overrides


def _from_override(source_field: str, folded: str, cleaned: str, override: Override) -> StringMatch:
    if override.disposition == DISPOSITION_NOT_A_DRUG:
        return StringMatch(
            source_field=source_field,
            folded=folded,
            cleaned=cleaned,
            method=METHOD_OVERRIDE_NOT_A_DRUG,
            components_matched=0,
        )
    rxcui = str(override.ingredient_rxcui)
    concept = client.Concept(rxcui=rxcui, name=override.ingredient_name, tty="IN")
    return StringMatch(
        source_field=source_field,
        folded=folded,
        cleaned=cleaned,
        method=METHOD_OVERRIDE,
        rxcui=rxcui,
        ingredients=(concept,),
        components_matched=1,
    )


def match_string(
    source_field: str,
    raw: str,
    *,
    overrides: dict[str, Override] | None = None,
    settings: Settings | None = None,
) -> StringMatch:
    """Resolve one source string, trying each rung in order."""
    settings = settings or get_settings()
    overrides = overrides if overrides is not None else {}
    prepared = clean(raw)

    override = overrides.get(prepared.folded)
    if override is not None:
        return _from_override(source_field, prepared.folded, prepared.cleaned, override)

    ingredients: list[client.Concept] = []
    seen: set[str] = set()
    requests = cache_hits = matched_components = 0
    used_approximate = False
    single_rxcui: str | None = None
    score: float | None = None
    rank: int | None = None

    for component in prepared.components:
        resolution = client.resolve_exact(component, settings)
        if resolution is None:
            resolution = client.resolve(component, settings)
            if resolution.matched:
                used_approximate = True
        requests += resolution.requests
        cache_hits += resolution.cache_hits
        if not resolution.matched:
            continue

        matched_components += 1
        single_rxcui = resolution.rxcui
        score = resolution.score if resolution.score is not None else score
        rank = resolution.rank if resolution.rank is not None else rank
        for concept in resolution.ingredients:
            if concept.rxcui not in seen:
                seen.add(concept.rxcui)
                ingredients.append(concept)

    if not ingredients:
        method = METHOD_UNMATCHED
    elif used_approximate:
        method = METHOD_APPROXIMATE
    else:
        method = METHOD_EXACT

    return StringMatch(
        source_field=source_field,
        folded=prepared.folded,
        cleaned=prepared.cleaned,
        method=method,
        # A combination resolves to several concepts and therefore to no single
        # one. Recording the last component's would name a part as the whole.
        rxcui=single_rxcui if len(prepared.components) == 1 else None,
        ingredients=tuple(ingredients),
        score=score,
        rank=rank,
        components_total=len(prepared.components),
        components_matched=matched_components,
        requests=requests,
        cache_hits=cache_hits,
        steps=prepared.steps,
    )
