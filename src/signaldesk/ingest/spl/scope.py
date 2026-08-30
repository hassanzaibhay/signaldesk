"""Which drugs get labels fetched, and what to ask openFDA for them.

## The set

The signal table is keyed on raw drug strings - both published runs used
``drug_key: raw_string``, and ``analytics.contingency`` groups on
``upper(trim(drugname_raw))``. So "the drugs that carry signal" is a set of
strings like ``LIPITOR 10MG``, not a set of concepts, and this module is the
tracked definition of that set:

    A raw string carries signal when, for a named run, it appears in at least
    one scored pair where ``flag_ror_prr_bcpnn AND NOT insufficient``.

``flag_all_four`` is deliberately not used. MGPS converged onto a bound in both
published runs, the artifact marks EBGM05 provisional, and scoping a document
corpus on a column that must not be quoted would make the corpus unquotable too.

## The cap

The full flagged set is not fetched. ``select`` takes ``top_k`` and returns the
strings with the most qualifying pairs, ties broken by the string ascending so
the selection is deterministic and a re-run asks for the same labels. The point
is to measure the hit rate on a small set before deciding whether the scope is
worth widening.

## The query

openFDA cannot be searched for ``LIPITOR 10MG``. Two routes, in priority order:

1. the RxNorm ingredient rxcui for the string, if normalization has produced
   one;
2. a deterministically cleaned form of the string, matched against brand names.

Route 1 is correct and is kept, but it is currently dead: the RxNav cache is
lost and the normalization track is not being rebuilt, so ``DrugStringMatch`` is
empty and every string takes route 2. Every cost figure recorded for this
pipeline is therefore computed on route 2 alone - one request per string, with
none of the collapsing onto shared ingredients that route 1 would have given.

A curated override file supplies the query for strings the cleaner cannot
handle. It is human-authored; nothing here writes it.
"""

from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from signaldesk.core.config import Settings, get_settings
from signaldesk.core.errors import SignalDeskError
from signaldesk.core.logging import get_logger

log = get_logger(__name__)

#: Repository root, from this module: spl -> ingest -> signaldesk -> src.
REPO_ROOT = Path(__file__).resolve().parents[4]

#: Curated string-to-query overrides. Hassan's file. Absent is a valid state and
#: means no overrides, not an error.
OVERRIDE_PATH = REPO_ROOT / "data" / "overrides" / "spl_query.csv"

DEFAULT_TOP_K = 200


class SignalScopeError(SignalDeskError):
    """The signal table needed to define the scope is absent or unusable."""


@dataclass(frozen=True, slots=True)
class ScopeUnit:
    """One drug string to fetch labels for, and how it will be asked for."""

    folded_string: str
    query: str
    route: str
    ingredient_rxcui: int | None
    #: Qualifying flagged pairs behind this string. What the ranking sorted on.
    flagged_pairs: int

    @property
    def manifest_unit(self) -> str:
        """Key for the ingest manifest.

        ``IngestManifest.unit`` is 32 characters and a drug string is not, so a
        string-routed unit is keyed on a digest of the string rather than the
        string itself. The digest is of the folded string, which is what the
        selection is keyed on, so the unit is stable across runs.
        """
        if self.ingredient_rxcui is not None:
            return f"rxcui:{self.ingredient_rxcui}"
        digest = hashlib.sha256(self.folded_string.encode("utf-8")).hexdigest()
        return f"str:{digest[:24]}"


def signal_root(settings: Settings | None = None) -> Path:
    """Where the signal build writes its scored pairs."""
    settings = settings or get_settings()
    return settings.data_dir / "parquet" / "signal"


def latest_run_id(settings: Settings | None = None) -> str | None:
    """The most recent run partition, or None when none exist."""
    root = signal_root(settings)
    if not root.is_dir():
        return None
    partitions = sorted(p.name.removeprefix("run_id=") for p in root.glob("run_id=*"))
    return partitions[-1] if partitions else None


def _partition(run_id: str | None, settings: Settings | None) -> Path:
    """The partition for ``run_id``, or raise naming exactly what is missing.

    Raising rather than returning an empty frame is the whole point. The corpus
    is rebuilt from scratch periodically, and during a rebuild this directory
    does not exist. A run that quietly produced zero units would look identical
    to a run that correctly found no flagged strings, and the measured N would
    be recorded as 0 in an artifact.
    """
    root = signal_root(settings)
    if not root.is_dir():
        message = (
            f"the signal table does not exist at {root}. The corpus has not been "
            "built on this machine, or the build is still running. Run "
            "'make build-signals' first; this is not a scope of zero drugs."
        )
        raise SignalScopeError(message)
    resolved = run_id or latest_run_id(settings)
    if resolved is None:
        message = f"{root} exists but holds no run partitions; the signal build wrote nothing"
        raise SignalScopeError(message)
    partition = root / f"run_id={resolved}"
    if not partition.is_dir() or not any(partition.glob("*.parquet")):
        message = f"signal run {resolved!r} has no parquet under {partition}"
        raise SignalScopeError(message)
    return partition


def flagged_pair_counts(
    run_id: str | None = None, settings: Settings | None = None
) -> pl.DataFrame:
    """Qualifying flagged pairs per drug string, descending.

    The predicate is ``flag_ror_prr_bcpnn AND NOT insufficient``.

    The second conjunct currently removes nothing. In
    ``signals_20260818T051858Z.json`` the PS,SS run reports
    ``ror_prr_bcpnn`` 1,393,815, ``ror_prr_bcpnn_including_insufficient``
    1,393,815 and ``bcpnn`` 1,393,815 - identical - so on that run the
    conjunction reduces to ``flag_bcpnn`` and the sufficiency filter is a no-op.
    It is written out anyway because the intent is that a signal must clear the
    minimum cell count, and a future run with different data need not have that
    coincidence. It must never be described as a filter that removed anything.
    """
    partition = _partition(run_id, settings)
    frame = pl.read_parquet(partition / "*.parquet")
    missing = {"drug", "flag_ror_prr_bcpnn", "insufficient"} - set(frame.columns)
    if missing:
        message = f"signal parquet at {partition} lacks {sorted(missing)}"
        raise SignalScopeError(message)
    return (
        frame.filter(pl.col("flag_ror_prr_bcpnn") & ~pl.col("insufficient"))
        .group_by("drug")
        .agg(pl.len().alias("flagged_pairs"))
        .sort(["flagged_pairs", "drug"], descending=[True, False])
    )


#: Tokens that are dose, strength, form or release qualifiers rather than name.
#: Stripped only from the end of a string, and only until a token that is not
#: one of these is reached, so an interior word is never removed.
_UNITS = (
    "MG",
    "MCG",
    "UG",
    "G",
    "GM",
    "KG",
    "ML",
    "L",
    "IU",
    "U",
    "UNIT",
    "UNITS",
    "MEQ",
    "MMOL",
    "PERCENT",
)
_FORMS = (
    "TABLET",
    "TABLETS",
    "TAB",
    "TABS",
    "CAPSULE",
    "CAPSULES",
    "CAP",
    "CAPS",
    "INJECTION",
    "INJECTABLE",
    "SOLUTION",
    "SUSPENSION",
    "SYRUP",
    "ELIXIR",
    "CREAM",
    "OINTMENT",
    "GEL",
    "LOTION",
    "PATCH",
    "SPRAY",
    "INHALER",
    "INHALATION",
    "DROPS",
    "SUPPOSITORY",
    "POWDER",
    "GRANULES",
    "FILM",
    "LOZENGE",
    "KIT",
    "VIAL",
    "AMPULE",
    "SYRINGE",
    "PEN",
)
_QUALIFIERS = ("ER", "XR", "SR", "DR", "XL", "CR", "LA", "IR", "ODT", "EC", "PO", "IV", "IM")

_DROPPABLE = frozenset(_UNITS + _FORMS + _QUALIFIERS)

#: A bare number, or a number fused to a unit: "10", "0.5", "10MG", "81MG".
_NUMERIC = re.compile(r"^\d+(\.\d+)?(/\d+(\.\d+)?)?[A-Z%]*$")

_PARENTHETICAL = re.compile(r"\([^)]*\)")
_PUNCTUATION = re.compile(r"[,;:]+")


def _is_droppable(token: str) -> bool:
    """Whether one trailing token is dose, strength, form or release notation.

    A token may be a compound joined by a slash - ``UNITS/ML``, ``MG/ML``,
    ``100/50`` - and is droppable only when every part is. That asymmetry is
    what keeps ``875MG/CLAVULANATE`` intact: the second part is an ingredient
    name, so the whole token stays even though the first part is a strength.
    """
    if "/" in token:
        parts = [part for part in token.split("/") if part]
        return bool(parts) and all(part in _DROPPABLE or _NUMERIC.match(part) for part in parts)
    return token in _DROPPABLE or bool(_NUMERIC.match(token))


def clean_query(folded_string: str) -> str:
    """Reduce a FAERS drug string to something openFDA can match on a brand name.

    Deterministic and conservative: parenthetical asides are dropped, then
    trailing dose, strength, form and release tokens are peeled off one at a
    time until a token that is not one of those is reached. Nothing in the
    interior is touched, and the first token is never removed, so a string that
    is all droppable tokens still yields a query rather than an empty string.

    This is a name-shortener, not a normalizer. It has no vocabulary and cannot
    know that HCL is a salt or that a word is a misspelling. Strings it gets
    wrong are what the curated override file is for.
    """
    text = _PARENTHETICAL.sub(" ", folded_string.upper())
    text = _PUNCTUATION.sub(" ", text)
    tokens = [token for token in text.split() if token]
    if not tokens:
        return ""
    while len(tokens) > 1 and _is_droppable(tokens[-1]):
        tokens.pop()
    return " ".join(tokens)


def load_overrides(path: Path | None = None) -> dict[str, str]:
    """Curated string-to-query pairs. Empty when the file is absent.

    Absent is normal and is not an error: the override file exists to correct
    what the cleaner gets wrong, and until someone has looked at a hit-rate
    artifact there is nothing to correct.
    """
    target = path or OVERRIDE_PATH
    if not target.is_file():
        log.info("spl.overrides.absent", path=str(target))
        return {}
    overrides: dict[str, str] = {}
    with target.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            folded = (row.get("folded_string") or "").strip().upper()
            query = (row.get("openfda_query") or "").strip()
            if folded and query:
                overrides[folded] = query
    log.info("spl.overrides.loaded", path=str(target), rows=len(overrides))
    return overrides


def _ingredient_rxcuis() -> dict[str, int]:
    """Folded string to ingredient rxcui, from the normalization tables.

    Resolution is tested as ``match_method != unmatched``, never as
    ``rxcui IS NOT NULL``: 48,826 multi-component matches carry a null rxcui on
    the match row while being perfectly resolved, and the null test silently
    reads them as failures.

    Expected to be empty. The RxNav cache is lost and normalization is not being
    re-run, so this returns nothing and every string takes the cleaned-string
    route. The join is kept because the tables and the code are correct and the
    day the mapping is rebuilt this starts working with no change here.
    """
    from signaldesk.web.signals.models import DrugStringMatch

    rows = (
        DrugStringMatch.objects.exclude(match_method=DrugStringMatch.Method.UNMATCHED)
        .filter(ingredients__isnull=False)
        .values_list("folded_string", "ingredients__ingredient_id")
    )
    mapping: dict[str, int] = {}
    for folded, rxcui in rows:
        if folded not in mapping and rxcui is not None:
            mapping[folded] = int(rxcui)
    return mapping


def select(
    run_id: str | None = None,
    *,
    top_k: int = DEFAULT_TOP_K,
    settings: Settings | None = None,
    overrides: dict[str, str] | None = None,
    rxcuis: dict[str, int] | None = None,
) -> tuple[list[ScopeUnit], int]:
    """The capped scope, and N - the full flagged population it was capped from.

    Returns ``(units, n_total)``. ``n_total`` is the measurement the artifact
    records and the number that decides whether the scope widens; it is reported
    whether or not the cap bites.
    """
    if top_k < 1:
        message = f"top_k must be at least 1, got {top_k}"
        raise SignalScopeError(message)

    counts = flagged_pair_counts(run_id, settings)
    n_total = counts.height
    resolved_overrides = load_overrides() if overrides is None else overrides
    resolved_rxcuis = _ingredient_rxcuis() if rxcuis is None else rxcuis

    units: list[ScopeUnit] = []
    for folded, pairs in counts.head(top_k).iter_rows():
        override = resolved_overrides.get(folded)
        rxcui = resolved_rxcuis.get(folded)
        if override is not None:
            query, route, rxcui_used = override, "override", None
        elif rxcui is not None:
            query, route, rxcui_used = str(rxcui), "ingredient", rxcui
        else:
            query, route, rxcui_used = clean_query(folded), "cleaned_string", None
        units.append(
            ScopeUnit(
                folded_string=folded,
                query=query,
                route=route,
                ingredient_rxcui=rxcui_used,
                flagged_pairs=int(pairs),
            )
        )

    log.info(
        "spl.scope.selected",
        n_total=n_total,
        top_k=top_k,
        selected=len(units),
        by_route={
            route: sum(1 for unit in units if unit.route == route)
            for route in ("override", "ingredient", "cleaned_string")
        },
    )
    return units, n_total
