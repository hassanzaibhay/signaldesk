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
    #: Pages over the wire, and pages off disk. Kept apart because a warm cache
    #: otherwise reads as network cost. Names follow ingest.rxnorm.client.
    requests: int = 0
    cache_hits: int = 0
    network_bytes: int = 0
    cache_bytes: int = 0
    #: The label set was truncated at client.MAX_PAGES.
    #:
    #: None means unknown, and is not the same as False. A unit that fetched
    #: measures this. A unit that attached to another unit's fetch reads it back
    #: from the manifest, and a manifest row written before the flag existed
    #: yields None. Defaulting an attached unit to False would put 190 of 200 rows
    #: in the 06:05 artifact asserting no truncation without measuring it, three
    #: of them wrongly.
    page_cap: bool | None = False
    #: Reach-through rows written. Non-zero with documents at 0 means this
    #: string attached to a query another string fetched.
    drug_keys: int = 0
    #: set_id -> section count for the documents this unit wrote. The run's
    #: distinct totals are merged from these rather than counted off the tables,
    #: which would fold in rows written by earlier runs.
    document_sections: dict[str, int] = field(default_factory=dict)
    skipped: bool = False
    reason: str = ""
    error: str = ""
    #: False when nothing can be concluded about whether this string reaches a
    #: label: the unit errored, or it skipped against a manifest row that
    #: recorded no document count. Kept as a field rather than inferred from
    #: ``reason``, so the resolution rate does not depend on matching a sentence.
    terminal_state_known: bool = True

    @property
    def hit(self) -> bool:
        """This unit's own fetch found labels. Says nothing about a skipped unit."""
        return self.labels > 0

    @property
    def reaches_label(self) -> bool:
        """Terminal state: at least one label is reachable from this string now.

        Two ways to be true and they are not interchangeable. ``labels`` means
        this unit fetched them. ``drug_keys`` means another unit fetched them and
        this one attached its own reach-through, which is the normal outcome for
        every member of a query group but the first, and for every unit at all on
        a warm re-run.

        ``hit`` alone measures what this run did. This measures what is true when
        the run finishes, which is what a resolution rate is asking about.
        """
        return self.labels > 0 or self.drug_keys > 0

    @property
    def pages(self) -> int:
        return self.requests + self.cache_hits


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


def _attach_only(result: UnitResult, unit: ScopeUnit, decision: manifest.Decision) -> UnitResult:
    """A unit whose query another string already fetched.

    The fetch is skipped, but the string still gets its own drug keys: the
    manifest is keyed on the query and several strings can share one, and a
    retrieval lookup on the exact FAERS string has to resolve.

    Four outcomes, kept apart on purpose:

    * keys attached - the normal duplicate;
    * the manifest recorded zero documents, so there was nothing to attach;
    * the manifest recorded no count at all, so nothing is concluded and its
      own reason stands;
    * the manifest records documents but none can be found - an inconsistency,
      not an outcome. That one is an error, because a skip that quietly loses a
      string's reach-through is exactly the class of silent failure this whole
      change exists to remove.
    """
    result.skipped = True
    # Carried from the manifest row, never defaulted. This unit did not page, so
    # it cannot measure truncation; the unit that fetched recorded it. None when
    # the row predates the flag, and None must survive to the artifact as null.
    result.page_cap = decision.page_capped
    attached = store.attach_drug_keys(
        folded_string=unit.folded_string,
        query=unit.query,
        route=unit.route,
        ingredient_rxcui=unit.ingredient_rxcui,
    )
    result.drug_keys = attached
    recorded = decision.documents_recorded

    if attached:
        result.reason = "query already fetched; drug keys attached"
    elif recorded == 0:
        # The fetching unit found nothing. Expected, and the reason says so
        # rather than leaving a bare "already completed" to be misread.
        result.reason = "query already fetched and resolved to nothing"
    elif recorded is None:
        # No count on the row, so nothing can be concluded. The decision's own
        # reason is kept rather than inventing one: claiming "resolved to
        # nothing" here would assert something unknown. The same unknown is
        # flagged for the resolution rate, which must not read it as a miss.
        result.reason = decision.reason
        result.terminal_state_known = False
    else:
        result.skipped = False
        result.terminal_state_known = False
        result.error = (
            f"manifest records {recorded} documents for query {unit.query!r} "
            f"but no drug keys were found to attach to"
        )
        log.error(
            "spl.unit.attach_found_nothing",
            unit=unit.manifest_unit,
            query=unit.query,
            documents_recorded=recorded,
        )
    return result


def ingest_unit(unit: ScopeUnit, settings: Settings, *, force: bool = False) -> UnitResult:
    """Fetch, parse and store one drug string's labels."""
    result = UnitResult(unit=unit)
    decision = manifest.decide(unit.manifest_unit, force=force)
    if not decision.should_ingest:
        return _attach_only(result, unit, decision)

    if not unit.query:
        result.error = "the cleaner produced an empty query"
        result.terminal_state_known = False
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
        result.terminal_state_known = False
        log.error("spl.unit.failed", unit=unit.manifest_unit, error=repr(error))
        return result

    result.labels = counts.documents
    result.sections = counts.sections
    result.requests = found.requests
    result.cache_hits = found.cache_hits
    result.network_bytes = found.network_bytes
    result.cache_bytes = found.cache_bytes
    result.page_cap = found.page_cap
    # parse_results already deduplicates on set_id, so this is one entry per
    # distinct document. store_labels replaces a document's sections wholesale,
    # so a later unit writing the same set_id supersedes this count.
    result.document_sections = {record.set_id: len(record.sections) for record in records}
    manifest.complete(
        unit.manifest_unit,
        checksum=_checksum(records),
        row_counts={"documents": counts.documents, "sections": counts.sections},
        bytes_downloaded=found.network_bytes,
        page_cap=found.page_cap,
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

    # Terminal state, matching the artifact. Counting item.hit here reported one
    # of one on a warm re-run, because only the unit that fetched has labels of
    # its own and every other unit had attached to an earlier run's fetch.
    log.info(
        "spl.run.finished",
        n_total=n_total,
        selected=len(units),
        reaching_a_label=sum(1 for item in result.units if item.reaches_label),
        seconds=round(result.seconds, 1),
    )
    return result


def _distinct_documents(result: RunResult) -> dict[str, int]:
    """set_id -> section count across the whole run, later writes winning.

    Merged rather than summed. A document reached from several drug strings is
    one document and one set of sections; summing the per-unit counts is what
    made the old ``stored`` block report key rows as documents.
    """
    merged: dict[str, int] = {}
    for item in result.units:
        merged.update(item.document_sections)
    return merged


def _by_query(result: RunResult) -> dict[str, list[UnitResult]]:
    """Every selected unit, grouped by the query actually sent to openFDA.

    This is the run's definition of a distinct drug, and it is derived rather
    than declared: two selected strings that send one query are one fetch and
    are counted once. A hand-maintained rule would have needed editing the day
    the trailing period was folded, and again at the next cleaner change.

    Over ``result.units``, not ``result.attempted``. The scope is what the cap
    selected, and it does not shrink because a unit was skipped as already
    complete. Grouping the attempted units made the denominator a property of how
    warm the cache was: the 06:05 re-run attempted one unit, so it divided by one
    and reported both rates as 1.0 for a scope of 200 strings.
    """
    grouped: dict[str, list[UnitResult]] = {}
    for item in result.units:
        grouped.setdefault(item.unit.query, []).append(item)
    return grouped


def _terminal_state(group: list[UnitResult]) -> bool | None:
    """Whether this query reaches at least one label now. None when unknowable.

    True as soon as any member reaches one, however it got there. False when
    every member is accounted for and none does. None when no member reaches a
    label and at least one cannot say - a unit that errored, or one that skipped
    against a manifest row carrying no recorded count.

    The third case exists so it can be counted rather than absorbed. Folding an
    unknown into False depresses the rate by an amount nobody can see, which is
    the failure this rate has already had once.
    """
    if any(item.reaches_label for item in group):
        return True
    if all(item.terminal_state_known for item in group):
        return False
    return None


@dataclass(frozen=True, slots=True)
class Resolution:
    """The two rates and the counts behind them, computed once.

    Exists so the artifact and the terminal summary cannot disagree. The command
    line used to divide its own hits by its own attempted count, which is how it
    came to print "1 of 1 (100.0%)" beside an artifact describing 200 strings.

    Both denominators are properties of the selected scope. Neither moves because
    a unit was skipped, and neither is written down as a literal.
    """

    distinct_queries_selected: int
    distinct_queries_resolved: int
    distinct_queries_unknown_state: int
    selected_slots: int

    @property
    def hit_rate_distinct_queries(self) -> float | None:
        """Resolved queries over queries selected. None only for an empty scope."""
        if not self.distinct_queries_selected:
            return None
        return round(self.distinct_queries_resolved / self.distinct_queries_selected, 4)

    @property
    def hit_rate_selected_slots(self) -> float | None:
        """The same numerator over the slots top_k bought.

        The numerator is resolved *queries*, not resolved slots, and the
        difference is not cosmetic. Counting slots that reach a label gives
        191/200 on the measured scope, which is the 95.5% that double-counted
        query twins and has been retired.
        """
        if not self.selected_slots:
            return None
        return round(self.distinct_queries_resolved / self.selected_slots, 4)


def resolution(result: RunResult) -> Resolution:
    """Terminal-state resolution over the whole selected scope."""
    states = [_terminal_state(group) for group in _by_query(result).values()]
    return Resolution(
        distinct_queries_selected=len(states),
        distinct_queries_resolved=sum(1 for state in states if state is True),
        distinct_queries_unknown_state=sum(1 for state in states if state is None),
        selected_slots=len(result.units),
    )


def _distinct_cache_entries(result: RunResult) -> int:
    """Cache entries touched, as distinct from cache reads.

    Every member of a query group pages through the same query, so under
    ``--force`` a group of two reads each of its pages twice. ``cache_hits``
    counts the reads; this counts the entries behind them, taken as the largest
    page count in each group since the members page identically.

    It exists to keep two numbers apart that are equal by coincidence on this
    corpus. The clean run reads 318 pages from cache and there are 318 files in
    the cache directory, and those are unrelated quantities: 327 reads minus 9
    that go to the network, against 317 entries written keyless by the 10:16 run
    plus one written with a key. Only 261 entries are actually read. Reporting
    the third number makes it impossible to read either of the first two as
    evidence for the other.
    """
    return sum(
        max((item.cache_hits for item in group), default=0) for group in _by_query(result).values()
    )


def _unit_row(item: UnitResult) -> dict[str, object]:
    """One unit's line in the artifact.

    Every fact a reader needs to check a claim about this unit without the log:
    what was asked, what came back, what it cost, and whether the answer was
    truncated.
    """
    row: dict[str, object] = {
        "folded_string": item.unit.folded_string,
        "query": item.unit.query,
        "route": item.unit.route,
        "flagged_pairs": item.unit.flagged_pairs,
        "documents": item.labels,
        "sections": item.sections,
        "drug_keys": item.drug_keys,
        "requests": item.requests,
        "cache_hits": item.cache_hits,
        "network_bytes": item.network_bytes,
        "cache_bytes": item.cache_bytes,
        "page_cap": item.page_cap,
    }
    if item.skipped:
        row["skipped"] = True
        row["reason"] = item.reason
    if item.error:
        row["error"] = item.error
    return row


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
    merged = _distinct_documents(result)
    resolved = resolution(result)

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
            "distinct_queries_selected": resolved.distinct_queries_selected,
            "distinct_queries_resolved": resolved.distinct_queries_resolved,
            "distinct_queries_unknown_state": resolved.distinct_queries_unknown_state,
            "hit_rate_distinct_queries": resolved.hit_rate_distinct_queries,
            "hit_rate_selected_slots": resolved.hit_rate_selected_slots,
            "rate_note": (
                "Two rates, one numerator, and both are needed. "
                "hit_rate_distinct_queries groups units by the query sent to "
                "openFDA, so two strings that send one query count once; it "
                "measures whether the query strategy works. "
                "hit_rate_selected_slots divides the same numerator by the "
                "slots top_k bought; it measures what the cap actually "
                "delivered. The gap between them is the duplication cost. "
                "Neither supersedes the other and neither should be quoted "
                "alone. "
                "Both are computed over the whole selected scope from terminal "
                "state, never over this run's attempts. A string reaches a label "
                "whether this run fetched it or attached to a fetch an earlier "
                "run made, so a warm re-run that attempts one unit reports the "
                "same rates as the cold run that fetched all of them. "
                "The numerator is distinct resolving queries in both. Dividing "
                "resolving slots by slots instead would give 191/200 on this "
                "scope, which is the double-counted 95.5% already retired."
            ),
            "unknown_state_note": (
                "distinct_queries_unknown_state counts queries whose terminal "
                "state could not be read: every member either errored or skipped "
                "against a manifest row recording no document count. They are in "
                "both denominators and in neither numerator, so they depress the "
                "rates rather than shrinking the scope. Non-zero here means the "
                "rates are lower bounds and the cause is worth finding."
            ),
            "resolved_to_nothing": len(attempted) - len(hits),
            "from_override_file": len(overridden),
            "by_route": {
                route: len(result.by_route(route))
                for route in ("override", "ingredient", "cleaned_string")
            },
            "skipped_already_complete": sum(1 for item in result.units if item.skipped),
            "attached_to_another_units_fetch": sum(
                1 for item in result.units if item.skipped and item.drug_keys
            ),
            "errors": sum(1 for item in result.units if item.error),
        },
        "stored": {
            "drug_key_rows": sum(item.labels for item in result.units),
            "section_writes": sum(item.sections for item in result.units),
            "distinct_documents": len(merged),
            "distinct_sections": sum(merged.values()),
            "note": (
                "drug_key_rows counts one (drug string, document) row, and "
                "section_writes one write, so a label reached from several "
                "strings is counted once per string. The distinct_* figures "
                "count the label. Both are reported because the ratio between "
                "them is the duplication in the scope, not an error in either."
            ),
        },
        "cost": {
            "network_requests": sum(item.requests for item in result.units),
            "cache_hits": sum(item.cache_hits for item in result.units),
            "distinct_cache_entries_read": _distinct_cache_entries(result),
            "pages_total": sum(item.pages for item in result.units),
            "network_bytes": sum(item.network_bytes for item in result.units),
            "cache_bytes": sum(item.cache_bytes for item in result.units),
            "seconds": round(result.seconds, 2),
            "daily_cap_assumed": client.daily_cap(settings),
            "api_key_present": bool(settings.openfda_api_key),
            "note": (
                "network_requests is pages that left the machine; cache_hits is "
                "pages served from the on-disk cache, which cost no request and "
                "no rate limiting. Only network_requests counts against the "
                "daily cap. "
                "cache_hits counts reads, distinct_cache_entries_read counts the "
                "entries behind them, and neither is the number of files in the "
                "cache directory. Under --force every member of a query group "
                "pages through the same query, so reads exceed entries by the "
                "duplication in the scope. Do not read one as evidence for "
                "another. "
                "Bytes are split the same way, so network_bytes is "
                "transfer and cache_bytes is replay. The ingredient route, "
                "which would have collapsed many strings onto one request per "
                "shared ingredient, contributes nothing: RxNorm normalization is "
                "not being rebuilt and DrugStringMatch is empty, so every "
                "distinct query costs its own request. "
                "A re-run after a cleaner change is not a cold run: every query "
                "the current cleaner emits may already have been sent, in which "
                "case it is served from the on-disk cache and costs no request. "
                "Expect a small non-zero network count even then, because only "
                "200 responses are cached and a query that returned 404 is "
                "re-sent every time. Read a low network_requests as a property "
                "of a warm re-run, never as the cost of this ingest from cold."
            ),
        },
        "quotable": {
            "note": (
                "hit_rate_distinct_queries and hit_rate_selected_slots are "
                "measured and quotable, together and never one alone. "
                "n_total_flagged_strings is measured and quotable, and "
                "supersedes any earlier 'head drugs' figure, which had no "
                "tracked definition. distinct_documents, not drug_key_rows, is "
                "the size of the corpus."
            ),
        },
        "units_note": (
            "One row per selected unit, in selection order, which is the "
            "ranking: descending flagged pairs, ties broken by the string "
            "ascending. A row with drug_keys above zero and documents at zero "
            "did not resolve to nothing - it shares a query with another "
            "string, which fetched the labels, and this row attached its own "
            "reach-through to them. The documents and sections are counted "
            "once, against the unit that fetched them. "
            "page_cap is true, false, or null, and null is not false. A unit "
            "that fetched measured it. A unit that attached carries it over from "
            "the manifest row of the unit that fetched, and null means that row "
            "predates the flag being recorded, so truncation is unknown. A "
            "consumer excluding page-capped strings must exclude null too: it is "
            "an absence of measurement, not a measurement of absence."
        ),
        # Selection order, which is the ranking: descending flagged pairs, ties
        # broken by the string ascending. Deterministic, so two runs of the same
        # scope diff cleanly and a per-unit fact can be checked without the log.
        "units": [_unit_row(item) for item in result.units],
    }


def write_artifact(result: RunResult, root: Path | None = None) -> Path:
    """Write the run artifact and return its path."""
    directory = root or history_root()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"spl_ingest_{result.run_id}.json"
    path.write_text(json.dumps(artifact(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log.info("spl.artifact.written", path=str(path))
    return path
