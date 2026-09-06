"""Paged, filtered reads of one signal run, straight out of Parquet.

The signal table is 8,583,614 scored pairs in a single 928 MB Parquet file, of
which 2,785,896 clear the minimum cell count. It is not in Postgres and this
module does not put it there. Every request opens a private in-memory DuckDB,
projects the columns the page actually shows, pushes the filter and the ordering
into the engine, and takes a page off the top.

Why that is fast enough, measured in the running container against the real
partition, warm, at four threads:

===========================================  ========
Query                                        Latency
===========================================  ========
count(*) where a >= 3, from zone maps           3 ms
first page, sort a desc, 17 columns            20 ms
first page, sort ror desc                      89 ms
drug and event contains-filter, page + sort    62 ms
filtered count                              9 - 14 ms
filter matching nothing                         9 ms
chart payload, top 500                         34 ms
OFFSET 2,785,800, full sort                   610 ms
cold connection plus deepest offset           520 ms
===========================================  ========

Peak resident memory across that whole sequence, including the 2.7 million row
sort, was 416 MB against a 928 MB file. Parquet is columnar so only the
projected columns are decompressed, and ``ORDER BY ... LIMIT`` is a bounded
top-N heap rather than a materialised sort. The corpus is never in memory.

Two properties are load-bearing and are enforced rather than documented:

* The connection is in-memory. ``core.db.connect`` takes an exclusive file lock
  on the persistent database, which would serialise concurrent requests and
  leave a stale lock behind a killed one. Nothing here writes, so nothing here
  takes that lock.
* User text reaches DuckDB as a bound parameter and never as string
  interpolation, and the sort column is an enum member whose value is the only
  thing that reaches the SQL. A sort column cannot be supplied by a request.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

import duckdb
from pydantic import BaseModel, ConfigDict, Field, field_validator

from signaldesk.analytics.signals import latest_run, signal_root
from signaldesk.core.config import Settings, get_settings
from signaldesk.core.db import connect
from signaldesk.core.errors import SignalDeskError
from signaldesk.core.logging import get_logger

log = get_logger(__name__)

#: The run this page was built and checked against. Its record is committed at
#: evals/history/signals_20260831T091650Z.json, and the banner numbers on the
#: page are checked against that file. Pinned rather than always-latest so that
#: a new build does not silently change what the published figures describe.
PINNED_RUN_ID: Final = "20260831T090758Z"

#: ARCHITECTURE section 8.1: a pair below this many co-reported cases is not a
#: signal, whatever the estimators say. The run itself was built with min_a 3,
#: recorded in the artifact as ``params.min_a``.
MIN_A: Final = 3

#: Rows per page. Fixed rather than request-controlled: the page size is what
#: the latency table above was measured at, and letting a request ask for
#: 100,000 rows would hand it the whole table one request at a time.
PAGE_SIZE: Final = 50

#: Points in the scatter. The cap is printed on the chart itself, never implied
#: away, because a 1,000-point view of 2.7 million pairs read as the whole
#: corpus would be a lie told by omission.
CHART_POINTS: Final = 1000

#: Bounded so the engine cannot take the container's headroom for one page
#: request. ``core.db.connect`` defaults to 8 GB, which is right for a corpus
#: build and wrong for a web request; the deepest page measured at 416 MB.
WEB_MEMORY_LIMIT: Final = "1GB"
WEB_THREADS: Final = 2

#: Columns read for the table. Named rather than ``*``: the projection is what
#: keeps a page off the 22 columns it does not show.
_ROW_COLUMNS: Final = (
    "drug",
    "pt",
    "a",
    "ror",
    "ror_lower",
    "ror_upper",
    "prr",
    "prr_lower",
    "prr_upper",
    "ic025",
    "corrected",
    "flag_ror",
    "flag_prr",
    "flag_bcpnn",
    "flag_ror_prr_bcpnn",
)


class SignalQueryError(SignalDeskError):
    """The signal run backing the page is absent or unreadable."""


class SortColumn(StrEnum):
    """The columns a request may order by.

    An enum, not a string, and the membership check is what stands between a
    query parameter and the ``ORDER BY`` clause. EBGM and EBGM05 are absent on
    purpose: the gamma prior converged onto a bound on this run, so sorting by
    them would rank the table on a number the run does not support.
    """

    DRUG = "drug"
    PT = "pt"
    A = "a"
    ROR = "ror"
    PRR = "prr"
    IC025 = "ic025"


class SortDirection(StrEnum):
    """Ordering direction."""

    ASC = "asc"
    DESC = "desc"

    @property
    def sql(self) -> str:
        return "ASC" if self is SortDirection.ASC else "DESC"


#: What each sortable column is called in the interface. Kept next to the enum
#: so a new sort column cannot be added without naming it.
SORT_LABELS: Final[dict[SortColumn, str]] = {
    SortColumn.DRUG: "Drug",
    SortColumn.PT: "Event",
    SortColumn.A: "Cases",
    SortColumn.ROR: "ROR",
    SortColumn.PRR: "PRR",
    SortColumn.IC025: "IC025",
}


class SignalQuery(BaseModel):
    """One request's filter, ordering and position.

    Built from query parameters, which means every field is attacker-controlled
    and every field is therefore validated here rather than at the point of use.
    """

    model_config = ConfigDict(frozen=True)

    drug: str = ""
    event: str = ""
    sort: SortColumn = SortColumn.A
    direction: SortDirection = SortDirection.DESC
    page: int = Field(default=1, ge=1)

    @field_validator("drug", "event")
    @classmethod
    def _fold(cls, value: str) -> str:
        """Fold a filter term the way the run's keys were folded.

        The signal run was built with ``drug_key=raw_string``, which groups on
        ``upper(trim(drugname_raw))``, and terms are folded the same way in
        ``analytics.contingency``. Matching a lower-case term against those
        keys would return nothing and look like an empty corpus.

        Truncated because the column is free text and a filter term longer than
        any value in it can only ever match nothing, slowly.
        """
        return value.strip().upper()[:120]

    @property
    def offset(self) -> int:
        return (self.page - 1) * PAGE_SIZE

    @property
    def is_filtered(self) -> bool:
        return bool(self.drug or self.event)

    def describe_filter(self) -> str:
        """The filter in words, for the chart caption and the empty state."""
        parts = []
        if self.drug:
            parts.append(f"drug contains {self.drug!r}")
        if self.event:
            parts.append(f"event contains {self.event!r}")
        return " and ".join(parts) if parts else "no filter"


@dataclass(frozen=True, slots=True)
class SignalRow:
    """One scored drug-event pair, as the table shows it.

    EBGM and EBGM05 are deliberately not fields. The MGPS column on the page
    carries no number for this run, and a row object that carried one would be
    an invitation to render it.
    """

    drug: str
    pt: str
    a: int
    ror: float | None
    ror_lower: float | None
    ror_upper: float | None
    prr: float | None
    prr_lower: float | None
    prr_upper: float | None
    ic025: float | None
    corrected: bool
    flag_ror: bool
    flag_prr: bool
    flag_bcpnn: bool
    flag_ror_prr_bcpnn: bool


@dataclass(frozen=True, slots=True)
class ChartPoint:
    """One point in the scatter."""

    drug: str
    pt: str
    a: int
    ic025: float
    flagged: bool


@dataclass(frozen=True, slots=True)
class SignalPage:
    """A page of rows, and everything the page needs to describe itself."""

    run_id: str
    rows: tuple[SignalRow, ...]
    total: int
    corpus_total: int
    query: SignalQuery

    @property
    def page_count(self) -> int:
        return max(1, -(-self.total // PAGE_SIZE))

    @property
    def has_previous(self) -> bool:
        return self.query.page > 1

    @property
    def has_next(self) -> bool:
        return self.query.page < self.page_count

    @property
    def first_index(self) -> int:
        """1-based index of the first row on this page, 0 when there are none."""
        return self.query.offset + 1 if self.rows else 0

    @property
    def last_index(self) -> int:
        return self.query.offset + len(self.rows)


def resolve_run_id(settings: Settings | None = None) -> str:
    """The run the page serves.

    The pinned run when it is on disk, otherwise the newest one there is. The
    fallback exists for machines that have built a corpus but not this one -
    tests and fresh clones - and it is never silent: the resolved id is logged
    here and printed in the page's provenance banner, so a page serving some
    other run says which one on its face.
    """
    root = signal_root(settings)
    if not root.is_dir():
        message = (
            f"the signal table does not exist at {root}. Nothing has been built on "
            "this machine. Run 'make build-signals'; this is not a corpus of zero pairs."
        )
        raise SignalQueryError(message)

    if (root / f"run={PINNED_RUN_ID}").is_dir():
        return PINNED_RUN_ID

    fallback = latest_run(settings)
    if fallback is None:
        message = f"no signal run partition under {root}; the directory exists and is empty"
        raise SignalQueryError(message)
    log.warning("signals.run.fallback", pinned=PINNED_RUN_ID, serving=fallback)
    return fallback


def _partition_glob(run_id: str, settings: Settings | None = None) -> str:
    return (signal_root(settings) / f"run={run_id}" / "*.parquet").as_posix()


def _contains(term: str) -> str:
    """A LIKE pattern matching ``term`` as a literal substring.

    ``%`` and ``_`` are wildcards in LIKE, and drug strings genuinely contain
    both - "MODULE_1" and "5% DEXTROSE" are the shape of thing in this column.
    Passing a term through unescaped would quietly turn a search for one of them
    into a search for anything, so the metacharacters and the escape character
    itself are escaped and the pattern declares its escape.
    """
    escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _where(query: SignalQuery) -> tuple[str, list[object]]:
    """The WHERE clause and its bound parameters.

    Every user-supplied value is a ``?``. The clause text is built from the
    presence of a term, never from its content.
    """
    clauses = [f"a >= {MIN_A}"]
    parameters: list[object] = []
    if query.drug:
        clauses.append("drug LIKE ? ESCAPE '\\'")
        parameters.append(_contains(query.drug))
    if query.event:
        clauses.append("pt LIKE ? ESCAPE '\\'")
        parameters.append(_contains(query.event))
    return " AND ".join(clauses), parameters


def _order_by(query: SignalQuery) -> str:
    """The ORDER BY clause.

    ``query.sort`` is a ``SortColumn``, so its value is one of six literals
    written in this module and cannot be anything else. Nulls sort last in both
    directions: a pair whose estimator did not produce a value is not the
    strongest signal in the table and is not the weakest either, it is unknown,
    and unknowns belong at the end. Drug and term break ties so that paging is
    stable across requests.
    """
    column = SortColumn(query.sort).value
    return f"{column} {query.direction.sql} NULLS LAST, drug ASC, pt ASC"


def _connect(settings: Settings | None = None) -> duckdb.DuckDBPyConnection:
    """A private, bounded, read-only-in-practice handle."""
    settings = settings or get_settings()
    handle = connect(settings, in_memory=True)
    handle.execute(f"SET memory_limit='{WEB_MEMORY_LIMIT}'")
    handle.execute(f"SET threads={WEB_THREADS}")
    return handle


def _row(record: tuple[Any, ...]) -> SignalRow:
    return SignalRow(
        drug=str(record[0]),
        pt=str(record[1]),
        a=int(record[2]),
        ror=_optional_float(record[3]),
        ror_lower=_optional_float(record[4]),
        ror_upper=_optional_float(record[5]),
        prr=_optional_float(record[6]),
        prr_lower=_optional_float(record[7]),
        prr_upper=_optional_float(record[8]),
        ic025=_optional_float(record[9]),
        corrected=bool(record[10]),
        flag_ror=bool(record[11]),
        flag_prr=bool(record[12]),
        flag_bcpnn=bool(record[13]),
        flag_ror_prr_bcpnn=bool(record[14]),
    )


def _optional_float(value: object) -> float | None:
    """A finite float, or None.

    A non-finite estimator is not a number the page can render, and formatting
    ``inf`` into a cell would put a value there that no reader can act on. None
    is the honest state and the template says so in words.
    """
    if value is None:
        return None
    number = float(value)  # type: ignore[arg-type]
    return number if number == number and abs(number) != float("inf") else None


def fetch_page(query: SignalQuery, settings: Settings | None = None) -> SignalPage:
    """One page of rows, with the filtered and unfiltered totals beside it.

    Both totals come from the same partition in the same connection as the rows,
    so the count under the table and the rows in it can never describe different
    data.
    """
    run_id = resolve_run_id(settings)
    source = f"read_parquet('{_partition_glob(run_id, settings)}')"
    where, parameters = _where(query)

    handle = _connect(settings)
    try:
        total = int(_scalar(handle, f"SELECT count(*) FROM {source} WHERE {where}", parameters))
        corpus_total = int(_scalar(handle, f"SELECT count(*) FROM {source} WHERE a >= {MIN_A}", []))
        records = handle.execute(
            f"SELECT {', '.join(_ROW_COLUMNS)} FROM {source} WHERE {where} "
            f"ORDER BY {_order_by(query)} LIMIT {PAGE_SIZE} OFFSET {query.offset}",
            parameters,
        ).fetchall()
    finally:
        handle.close()

    return SignalPage(
        run_id=run_id,
        rows=tuple(_row(record) for record in records),
        total=total,
        corpus_total=corpus_total,
        query=query,
    )


def fetch_chart_points(
    query: SignalQuery, settings: Settings | None = None, *, limit: int = CHART_POINTS
) -> tuple[ChartPoint, ...]:
    """The scatter's points: the pairs with the most cases under this filter.

    Ordered by ``a`` rather than by the table's current sort on purpose. Sorting
    the chart the way the table is sorted would, on a sort by IC025, hand back a
    thousand rows that all share roughly one IC025 and draw a flat band; the
    plot is meant to show the spread of the filtered population, not the top of
    the current page.
    """
    run_id = resolve_run_id(settings)
    source = f"read_parquet('{_partition_glob(run_id, settings)}')"
    where, parameters = _where(query)

    handle = _connect(settings)
    try:
        records = handle.execute(
            f"SELECT drug, pt, a, ic025, flag_ror_prr_bcpnn FROM {source} "
            f"WHERE {where} AND ic025 IS NOT NULL AND isfinite(ic025) "
            f"ORDER BY a DESC, drug ASC, pt ASC LIMIT {int(limit)}",
            parameters,
        ).fetchall()
    finally:
        handle.close()

    return tuple(
        ChartPoint(
            drug=str(record[0]),
            pt=str(record[1]),
            a=int(record[2]),
            ic025=float(record[3]),
            flagged=bool(record[4]),
        )
        for record in records
    )


def _scalar(
    handle: duckdb.DuckDBPyConnection, sql: str, parameters: list[object]
) -> int | float | str:
    result = handle.execute(sql, parameters).fetchone()
    if result is None:  # pragma: no cover - an aggregate always returns a row
        message = f"query returned no row: {sql}"
        raise SignalQueryError(message)
    value: int | float | str = result[0]
    return value
