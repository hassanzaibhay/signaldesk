"""Link building and the figure spec, without a corpus behind them.

The figure is asserted here rather than in a browser because what it has to
carry is textual: the point cap and the active filter have to be inside the
image, so that a screenshot of a filtered chart cannot be read as the whole
corpus.
"""

from __future__ import annotations

import pytest

from signaldesk.web.signals.provenance import LabelProvenance
from signaldesk.web.signals.query import ChartPoint, SignalPage, SignalQuery, SortColumn
from signaldesk.web.signals.views import _figure, _sort_links, _url

pytestmark = pytest.mark.unit


def _page(query: SignalQuery, total: int = 2_785_896) -> SignalPage:
    return SignalPage(
        run_id="20260831T090758Z", rows=(), total=total, corpus_total=2_785_896, query=query
    )


def test_the_default_view_has_a_bare_url() -> None:
    """A shared link is the short form of what it describes."""
    assert _url(SignalQuery()) == "?"


def test_only_values_that_differ_from_the_default_are_written() -> None:
    assert _url(SignalQuery(drug="aspirin")) == "?drug=ASPIRIN"


def test_overrides_win_over_the_current_query() -> None:
    url = _url(SignalQuery(drug="aspirin", page=4), page=1)

    assert url == "?drug=ASPIRIN"


def test_sorting_a_new_column_starts_descending_and_resets_the_page() -> None:
    """The useful end of every one of these scales is the top."""
    links = {link["column"]: link for link in _sort_links(SignalQuery(page=9))}

    assert links["ror"]["url"] == "?sort=ror"
    assert links["ror"]["active"] is False


def test_clicking_the_active_column_flips_the_direction() -> None:
    links = {link["column"]: link for link in _sort_links(SignalQuery(sort=SortColumn.A))}

    assert links["a"]["active"] is True
    assert links["a"]["direction"] == "desc"
    assert links["a"]["url"] == "?direction=asc"


def test_every_sortable_column_gets_a_link() -> None:
    links = _sort_links(SignalQuery())

    assert {link["column"] for link in links} == {column.value for column in SortColumn}


def _points() -> tuple[ChartPoint, ...]:
    return (
        ChartPoint(drug="ASPIRIN", pt="RASH", a=900, ic025=3.4, flagged=True),
        ChartPoint(drug="IBUPROFEN", pt="NAUSEA", a=120, ic025=2.1, flagged=False),
    )


def test_the_figure_states_its_point_cap_and_its_filter_inside_the_image() -> None:
    """A cropped screenshot must still say what it is a picture of."""
    query = SignalQuery(drug="aspirin")
    title = _figure(_points(), _page(query, total=3_684))["layout"]["title"]["text"]

    assert "2 of 3,684 matching pairs" in title
    assert "cap 1,000" in title
    assert "drug contains 'ASPIRIN'" in title
    assert "20260831T090758Z" in title


def test_an_unfiltered_figure_says_it_is_unfiltered() -> None:
    title = _figure(_points(), _page(SignalQuery()))["layout"]["title"]["text"]

    assert "Filter: no filter" in title


def test_flagged_and_unflagged_pairs_are_separate_traces() -> None:
    """One legend entry each, so the threshold line has something to divide."""
    data = _figure(_points(), _page(SignalQuery()))["data"]

    assert [trace["x"] for trace in data] == [[900], [120]]
    assert "flag" in data[0]["name"].lower()


def test_the_threshold_line_is_drawn_at_the_bcpnn_signal_rule() -> None:
    """The BCPNN rule is IC025 > 0. Drawing it beats asserting it in prose."""
    layout = _figure(_points(), _page(SignalQuery()))["layout"]

    assert layout["shapes"][0]["y0"] == 0
    assert layout["shapes"][0]["y1"] == 0
    assert "IC025 = 0" in layout["annotations"][0]["text"]


def test_the_figure_plots_no_mgps_quantity() -> None:
    """EBGM is withheld for this run and must not reach a chart axis either."""
    figure = _figure(_points(), _page(SignalQuery()))

    assert "ebgm" not in repr(figure).lower()


def test_the_two_label_hit_rates_are_only_available_together() -> None:
    """The artifact says they are quotable together and never one alone."""
    scope = LabelProvenance(
        artifact="spl_ingest_20260901T114909Z.json",
        run_id="20260901T114909Z",
        selected_slots=200,
        distinct_queries_selected=169,
        distinct_queries_resolved=160,
        hit_rate_distinct_queries=0.9467,
        hit_rate_selected_slots=0.8,
        label_carrying_strings=191,
        total_flagged_strings=30549,
        distinct_documents=13205,
        distinct_sections=30111,
    )

    sentence = scope.rates_sentence

    assert "160 of 169 distinct queries resolved (94.7 percent)" in sentence
    assert "160 of the 200 selected slots (80.0 percent)" in sentence
    # 191 is a count of strings. Dividing it by the slot count reproduces the
    # 95.5 percent the artifact explicitly retires as double-counted.
    assert "191" not in sentence
    assert "95.5" not in sentence
