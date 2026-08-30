"""Fetch labels for the capped signal-carrying scope, and record the hit rate.

The artifact this writes is the deliverable. Whether the scope is worth widening
past ``top_k`` is decided by one number - how many of the selected strings
resolved to at least one label - and that number does not exist anywhere until a
run produces it. Everything else here exists to produce it honestly:

* ``n_total`` is the full flagged population, reported whether or not the cap
  bit, so the artifact says what fraction of the population was sampled;
* misses are counted by cause, because "the cleaner produced a query openFDA
  does not know" and "this drug genuinely has no label" need different fixes and
  only the first is worth curating an override for;
* the run refuses to start rather than reporting a scope of zero when the signal
  table is absent.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from signaldesk.core.config import Settings, get_settings
from signaldesk.core.logging import get_logger
from signaldesk.ingest.spl import client, manifest, scope, store
from signaldesk.ingest.spl.parse import LabelRecord, parse_results
from signaldesk.ingest.spl.scope import ScopeUnit

log = get_logger(__name__)

#: Repository root, from this module: spl -> ingest -> signaldesk -> src.
REPO_ROOT = Path(__file__).resolve().parents[4]


def history_root() -> Path:
    """Where committed metric snapshots live. Repository data, not configurable."""
    return REPO_ROOT / "evals" / "history"


@dataclass(slots=True)
class UnitResult:
    """What one drug string's fetch produced."""

    unit: ScopeUnit
    labels: int = 0
    sections: int = 0
    requests: int = 0
    bytes_received: int = 0
    skipped: bool = False
    reason: str = ""
    error: str = ""

    @property
    def hit(self) -> bool:
        return self.labels > 0


@dataclass(slots=True)
class RunResult:
    """What one pass produced, and the numbers the artifact reports."""

    run_id: str
    n_total: int
    top_k: int
    units: list[UnitResult] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def attempted(self) -> list[UnitResult]:
        return [result for result in self.units if not result.skipped and not result.error]

    def by_route(self, route: str) -> list[UnitResult]:
        return [result for result in self.units if result.unit.route == route]


def _checksum(records: list[LabelRecord]) -> str:
    """A digest of what was stored, so a changed label is visible in the manifest."""
    payload = json.dumps(
        [[record.set_id, record.version, record.effective_time] for record in records],
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _fetch(unit: ScopeUnit, settings: Settings) -> client.SearchResult:
    if unit.route == "ingredient" and unit.ingredient_rxcui is not None:
        return client.by_rxcui(unit.ingredient_rxcui, settings)
    return client.by_brand_name(unit.query, settings)


def ingest_unit(unit: ScopeUnit, settings: Settings, *, force: bool = False) -> UnitResult:
    """Fetch, parse and store one drug string's labels."""
    result = UnitResult(unit=unit)
    decision = manifest.decide(unit.manifest_unit, force=force)
    if not decision.should_ingest:
        result.skipped = True
        result.reason = decision.reason
        return result

    if not unit.query:
        result.error = "the cleaner produced an empty query"
        return result

    manifest.start(unit.manifest_unit)
    try:
        found = _fetch(unit, settings)
        records = parse_results(found.results)
        counts = store.store_labels(
            records,
            folded_string=unit.folded_string,
            query=unit.query,
            route=unit.route,
            ingredient_rxcui=unit.ingredient_rxcui,
        )
    except Exception as error:
        # Deliberately broad, and deliberately not re-raised. One drug string
        # that openFDA answers strangely must not end a run over hundreds of
        # them. The failure is written to the manifest, so the next run retries
        # exactly this unit, and it is counted in the artifact so a run with a
        # wall of errors cannot read as a low hit rate.
        manifest.fail(unit.manifest_unit, repr(error))
        result.error = repr(error)
        log.error("spl.unit.failed", unit=unit.manifest_unit, error=repr(error))
        return result

    result.labels = counts.documents
    result.sections = counts.sections
    result.requests = found.requests_made
    result.bytes_received = found.bytes_received
    manifest.complete(
        unit.manifest_unit,
        checksum=_checksum(records),
        row_counts={"documents": counts.documents, "sections": counts.sections},
        bytes_downloaded=found.bytes_received,
    )
    return result


def run(
    *,
    run_id: str | None = None,
    top_k: int = scope.DEFAULT_TOP_K,
    force: bool = False,
    settings: Settings | None = None,
) -> RunResult:
    """Select the capped scope and fetch it.

    Raises ``SignalScopeError`` when the signal table is absent. That is
    deliberate: a run that returned an empty result would be indistinguishable
    from one that correctly found nothing, and the artifact would record N = 0
    as a measurement.
    """
    settings = settings or get_settings()
    units, n_total = scope.select(run_id, top_k=top_k, settings=settings)

    cap = client.daily_cap(settings)
    if not settings.openfda_api_key:
        log.warning(
            "spl.run.no_api_key",
            daily_cap=cap,
            units=len(units),
            detail="OPENFDA_API_KEY is unset; the cap is 1,000 requests per day",
        )
    if len(units) > cap:
        log.warning("spl.run.over_daily_cap", units=len(units), daily_cap=cap)

    started = time.monotonic()
    result = RunResult(
        run_id=datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ"), n_total=n_total, top_k=top_k
    )
    for unit in units:
        result.units.append(ingest_unit(unit, settings, force=force))
    result.seconds = time.monotonic() - started

    log.info(
        "spl.run.finished",
        n_total=n_total,
        selected=len(units),
        hits=sum(1 for item in result.units if item.hit),
        seconds=round(result.seconds, 1),
    )
    return result


def artifact(result: RunResult, settings: Settings | None = None) -> dict[str, object]:
    """The document written to ``evals/history/``.

    ``hit_rate`` is the number the scope decision turns on: the share of
    attempted strings that reached at least one label. It is computed over
    attempted units only, because a unit skipped as already-complete says
    nothing about whether the query strategy works.
    """
    settings = settings or get_settings()
    attempted = result.attempted
    hits = [item for item in attempted if item.hit]
    overridden = result.by_route("override")

    return {
        "written_at": datetime.now(tz=UTC).isoformat(),
        "run_id": result.run_id,
        "scope": {
            "n_total_flagged_strings": result.n_total,
            "top_k": result.top_k,
            "selected": len(result.units),
            "definition": (
                "A raw drug string carries signal when it appears in at least one "
                "scored pair with flag_ror_prr_bcpnn and not insufficient, for the "
                "signal run read. Selection is the top_k strings by descending "
                "count of such pairs, ties broken by the string ascending."
            ),
            "predicate_note": (
                "The 'not insufficient' conjunct removed nothing on the runs "
                "measured so far: signals_20260818T051858Z.json reports "
                "ror_prr_bcpnn 1393815, ror_prr_bcpnn_including_insufficient "
                "1393815 and bcpnn 1393815, all equal, so the predicate reduces "
                "to flag_bcpnn. It is retained for intent and must not be "
                "reported as a filter that excluded pairs."
            ),
        },
        "resolution": {
            "attempted": len(attempted),
            "hit_at_least_one_label": len(hits),
            "hit_rate": round(len(hits) / len(attempted), 4) if attempted else None,
            "resolved_to_nothing": len(attempted) - len(hits),
            "from_override_file": len(overridden),
            "by_route": {
                route: len(result.by_route(route))
                for route in ("override", "ingredient", "cleaned_string")
            },
            "skipped_already_complete": sum(1 for item in result.units if item.skipped),
            "errors": sum(1 for item in result.units if item.error),
        },
        "stored": {
            "documents": sum(item.labels for item in result.units),
            "sections": sum(item.sections for item in result.units),
        },
        "cost": {
            "requests": sum(item.requests for item in result.units),
            "bytes_received": sum(item.bytes_received for item in result.units),
            "seconds": round(result.seconds, 2),
            "daily_cap_assumed": client.daily_cap(settings),
            "api_key_present": bool(settings.openfda_api_key),
            "note": (
                "One request per drug string plus paging. The ingredient route, "
                "which would have collapsed many strings onto one request per "
                "shared ingredient, contributes nothing: RxNorm normalization is "
                "not being rebuilt and DrugStringMatch is empty, so every string "
                "costs its own request."
            ),
        },
        "quotable": {
            "note": (
                "hit_rate is measured and quotable. n_total_flagged_strings is "
                "measured and quotable, and supersedes any earlier 'head drugs' "
                "figure, which had no tracked definition."
            ),
        },
    }


def write_artifact(result: RunResult, root: Path | None = None) -> Path:
    """Write the run artifact and return its path."""
    directory = root or history_root()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"spl_ingest_{result.run_id}.json"
    path.write_text(json.dumps(artifact(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log.info("spl.artifact.written", path=str(path))
    return path
