"""Filtering, ordering and paging of the signal table.

Over a fixture partition written through ``signal_root``, so these tests are
pinned to the layout the build actually writes rather than to a path spelled out
here. Nothing touches Postgres: the signal table is Parquet and this module
reads it directly.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from pydantic import ValidationError

from signaldesk.analytics.signals import signal_root
from signaldesk.core.config import Settings
from signaldesk.web.signals.query import (
    PAGE_SIZE,
    SignalQuery,
    SignalQueryError,
    SortColumn,
    SortDirection,
    fetch_chart_points,
    fetch_page,
    resolve_run_id,
)

pytestmark = pytest.mark.unit

RUN = "20260101T000000Z"

#: Six pairs, chosen so that every assertion below has a reason to fail.
#: - ASPIRIN/RASH is the largest ``a`` and the strongest on every estimator.
#: - ASPIRIN 81MG/RASH shares a drug prefix, so a contains-filter must find both.
#: - IBUPROFEN/NAUSEA is mid-range.
#: - MONOBENZONE/PEMPHIGUS has no ROR at all, which is what NULLS LAST is for.
#: - "5% DEXTROSE"/RASH carries a LIKE metacharacter in its name.
#: - TRACE/RASH sits below the minimum cell count and must never be returned.
_ROWS = {
    "drug": ["ASPIRIN", "ASPIRIN 81MG", "IBUPROFEN", "MONOBENZONE", "5% DEXTROSE", "TRACE"],
    "pt": ["RASH", "RASH", "NAUSEA", "PEMPHIGUS", "RASH", "RASH"],
    "a": [900, 40, 120, 5, 12, 2],
    "ror": [12.5, 3.25, 6.0, None, 1.5, 99.0],
    "ror_lower": [11.0, 2.9, 5.4, None, 1.2, 50.0],
    "ror_upper": [14.0, 3.6, 6.7, None, 1.9, 150.0],
    "prr": [10.0, 3.0, 5.5, None, 1.4, 80.0],
    "prr_lower": [9.1, 2.7, 5.0, None, 1.1, 40.0],
    "prr_upper": [11.0, 3.3, 6.0, None, 1.8, 120.0],
    "ic025": [3.4, 1.2, 2.1, None, 0.1, 4.9],
    "corrected": [False, False, False, True, False, False],
    "flag_ror": [True, True, True, False, True, True],
    "flag_prr": [True, True, True, False, False, True],
    "flag_bcpnn": [True, True, True, False, True, True],
    "flag_ror_prr_bcpnn": [True, True, True, False, False, True],
}


@pytest.fixture
def corpus(tmp_path: Path, settings: Settings) -> Settings:
    """A settings object whose data directory holds one small signal run."""
    scoped = settings.model_copy(update={"data_dir": tmp_path / "data"})
    partition = signal_root(scoped) / f"run={RUN}"
    partition.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(_ROWS).write_parquet(partition / "part-0.parquet")
    return scoped


def _drugs(query: SignalQuery, corpus: Settings) -> list[str]:
    return [row.drug for row in fetch_page(query, corpus).rows]


def test_the_run_falls_back_to_what_is_on_disk_when_the_pinned_one_is_absent(
    corpus: Settings,
) -> None:
    """A machine that built its own corpus still gets a page, and is told which."""
    assert resolve_run_id(corpus) == RUN


def test_an_empty_data_directory_is_an_error_not_an_empty_corpus(
    tmp_path: Path, settings: Settings
) -> None:
    """Zero pairs and no table at all are different states and must look different."""
    scoped = settings.model_copy(update={"data_dir": tmp_path / "empty"})

    with pytest.raises(SignalQueryError, match="does not exist"):
        resolve_run_id(scoped)


def test_pairs_below_the_minimum_cell_count_are_never_served(corpus: Settings) -> None:
    """TRACE has the strongest ratios in the fixture and two cases. It is not a signal."""
    page = fetch_page(SignalQuery(), corpus)

    assert "TRACE" not in [row.drug for row in page.rows]
    assert page.total == 5
    assert page.corpus_total == 5


def test_a_lower_case_filter_matches_the_folded_keys(corpus: Settings) -> None:
    """The table is keyed on upper(trim(...)); a typed term is folded to match."""
    assert _drugs(SignalQuery(drug="aspirin"), corpus) == ["ASPIRIN", "ASPIRIN 81MG"]


def test_a_filter_term_is_matched_as_a_literal_substring(corpus: Settings) -> None:
    """A percent sign in a drug name is a character, not a LIKE wildcard."""
    assert _drugs(SignalQuery(drug="5%"), corpus) == ["5% DEXTROSE"]


def test_drug_and_event_filters_are_conjunctive(corpus: Settings) -> None:
    assert _drugs(SignalQuery(drug="ASPIRIN", event="NAUSEA"), corpus) == []


@pytest.mark.parametrize(
    ("column", "direction", "expected_first"),
    [
        (SortColumn.A, SortDirection.DESC, "ASPIRIN"),
        (SortColumn.A, SortDirection.ASC, "MONOBENZONE"),
        (SortColumn.ROR, SortDirection.DESC, "ASPIRIN"),
        (SortColumn.PRR, SortDirection.DESC, "ASPIRIN"),
        (SortColumn.IC025, SortDirection.DESC, "ASPIRIN"),
        (SortColumn.DRUG, SortDirection.ASC, "5% DEXTROSE"),
        (SortColumn.PT, SortDirection.ASC, "IBUPROFEN"),
    ],
)
def test_every_sortable_column_orders_the_page(
    corpus: Settings, column: SortColumn, direction: SortDirection, expected_first: str
) -> None:
    page = fetch_page(SignalQuery(sort=column, direction=direction), corpus)

    assert page.rows[0].drug == expected_first


@pytest.mark.parametrize("direction", [SortDirection.ASC, SortDirection.DESC])
def test_pairs_with_no_estimate_sort_last_in_both_directions(
    corpus: Settings, direction: SortDirection
) -> None:
    """An absent value is unknown, not smallest and not largest."""
    page = fetch_page(SignalQuery(sort=SortColumn.ROR, direction=direction), corpus)

    assert page.rows[-1].drug == "MONOBENZONE"
    assert page.rows[-1].ror is None


def test_a_sort_column_outside_the_allowlist_is_rejected() -> None:
    """The enum is the barrier between a query string and the ORDER BY clause."""
    with pytest.raises(ValidationError):
        SignalQuery.model_validate({"sort": "drug; DROP TABLE signal"})


def test_a_page_below_one_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SignalQuery.model_validate({"page": "0"})


def test_paging_past_the_end_returns_no_rows_and_says_so(corpus: Settings) -> None:
    page = fetch_page(SignalQuery(page=2), corpus)

    assert page.rows == ()
    assert page.total == 5
    assert page.has_previous is True
    assert page.has_next is False
    assert page.first_index == 0


def test_a_filter_matching_nothing_is_a_page_of_zero_not_an_error(corpus: Settings) -> None:
    """The empty state needs a total of zero and a page count of one to render."""
    page = fetch_page(SignalQuery(drug="NOTHINGLIKETHIS"), corpus)

    assert page.rows == ()
    assert page.total == 0
    assert page.corpus_total == 5
    assert page.page_count == 1
    assert page.has_next is False


def test_the_page_reports_its_own_position(corpus: Settings) -> None:
    page = fetch_page(SignalQuery(), corpus)

    assert page.page_count == 1
    assert page.first_index == 1
    assert page.last_index == 5
    assert len(page.rows) <= PAGE_SIZE


def test_the_chart_takes_the_pairs_with_the_most_cases_under_the_filter(
    corpus: Settings,
) -> None:
    """Ordered by cases whatever the table is sorted by, and capped."""
    points = fetch_chart_points(
        SignalQuery(sort=SortColumn.IC025, direction=SortDirection.ASC), corpus, limit=2
    )

    assert [point.drug for point in points] == ["ASPIRIN", "IBUPROFEN"]
    assert points[0].flagged is True


def test_the_chart_drops_pairs_with_no_ic025(corpus: Settings) -> None:
    """A point needs a y value. A pair without one is omitted, not plotted at zero."""
    points = fetch_chart_points(SignalQuery(), corpus)

    assert "MONOBENZONE" not in [point.drug for point in points]


def test_the_row_carries_no_mgps_value(corpus: Settings) -> None:
    """EBGM and EBGM05 are withheld for this corpus and are not on the row object."""
    row = fetch_page(SignalQuery(), corpus).rows[0]

    assert not hasattr(row, "ebgm")
    assert not hasattr(row, "ebgm05")


def test_the_filter_description_names_both_terms(corpus: Settings) -> None:
    query = SignalQuery(drug="aspirin", event="rash")

    assert query.describe_filter() == "drug contains 'ASPIRIN' and event contains 'RASH'"
    assert SignalQuery().describe_filter() == "no filter"
