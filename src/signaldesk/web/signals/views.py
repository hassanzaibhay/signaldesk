"""The signals explorer: one page over one committed signal run.

A single view serves both the whole page and the fragment HTMX swaps into it, so
there is exactly one place that decides what a filter means and one place that
decides what a page of rows is. There is no separate JSON endpoint: nothing
outside this page consumes these rows yet, and an API surface built before it
has a consumer is a surface to maintain for nobody.

Filtering, ordering and paging all happen in DuckDB against the run's Parquet.
Nothing is filtered client-side, because the set being filtered is 2,785,896
pairs and the browser is never given it.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlencode

from django.http import HttpRequest, HttpResponse
from django.shortcuts import render
from pydantic import ValidationError

from signaldesk.core.logging import get_logger
from signaldesk.web.signals import labels, provenance
from signaldesk.web.signals.query import (
    CHART_POINTS,
    MIN_A,
    SORT_LABELS,
    ChartPoint,
    SignalPage,
    SignalQuery,
    SortColumn,
    SortDirection,
    fetch_chart_points,
    fetch_page,
)

log = get_logger(__name__)

#: Slate 500 and slate 300, matching the stylesheet. Flagged pairs are the
#: darker mark: the eye should land on them first.
_FLAGGED_COLOUR = "#0f172a"
_UNFLAGGED_COLOUR = "#94a3b8"


def explorer(request: HttpRequest) -> HttpResponse:
    """The table, the chart and the counts, for one filter and one ordering."""
    query = _query_from(request)
    page = fetch_page(query)
    points = fetch_chart_points(query)
    label_states = labels.statuses_for(row.drug for row in page.rows)
    # Paired here rather than looked up in the template. A dictionary lookup
    # by key needs a custom filter in the Django template language, and a
    # filter that silently returns nothing for a missing key is how a row ends
    # up with a blank label cell - the one thing this column exists to avoid.
    table_rows = [{"row": row, "label": label_states[row.drug]} for row in page.rows]

    context = {
        "query": query,
        "page": page,
        "min_a": MIN_A,
        "table_rows": table_rows,
        "figure": _figure(points, page),
        "chart_points": len(points),
        "sort_links": _sort_links(query),
        "previous_url": _url(query, page=query.page - 1) if page.has_previous else "",
        "next_url": _url(query, page=query.page + 1) if page.has_next else "",
        "clear_url": _url(SignalQuery(sort=query.sort, direction=query.direction)),
        "run": provenance.run_provenance(page.run_id),
        "label_scope": provenance.label_provenance(),
    }

    # django-htmx sets this on the request. Read defensively so the view still
    # renders a whole page if the middleware is ever absent, rather than
    # returning a bare fragment with no chrome around it.
    if bool(getattr(request, "htmx", False)):
        return render(request, "signals/_results.html", context)
    return render(request, "signals/explorer.html", context)


def _query_from(request: HttpRequest) -> SignalQuery:
    """Validate the query string, falling back to defaults on anything invalid.

    A request can name a sort column that does not exist, or a negative page.
    Those are rejected here by never reaching the query: the fallback is the
    default view, not a 500 and not an interpolated string. The rejection is
    logged so a broken link is discoverable rather than silently ignored.
    """
    raw = {
        "drug": request.GET.get("drug", ""),
        "event": request.GET.get("event", ""),
        "sort": request.GET.get("sort", SortColumn.A.value),
        "direction": request.GET.get("direction", SortDirection.DESC.value),
        "page": request.GET.get("page", "1"),
    }
    try:
        return SignalQuery.model_validate(raw)
    except ValidationError as error:
        log.info("signals.query.rejected", errors=error.error_count(), path=request.path)
        return SignalQuery(drug=raw["drug"], event=raw["event"])


def _url(query: SignalQuery, **overrides: object) -> str:
    """The explorer URL for a variant of this query.

    Only non-default values are written, so a shared link is the short form of
    what it describes and the default view has a bare URL.
    """
    values: dict[str, object] = {
        "drug": query.drug,
        "event": query.event,
        "sort": query.sort.value,
        "direction": query.direction.value,
        "page": query.page,
    }
    values.update(overrides)
    defaults = SignalQuery()
    pairs = {
        key: value
        for key, value in values.items()
        if str(value) != str(getattr(defaults, key)) and value not in ("", None)
    }
    encoded = urlencode({key: str(value) for key, value in pairs.items()})
    return f"?{encoded}" if encoded else "?"


def _sort_links(query: SignalQuery) -> list[dict[str, object]]:
    """One entry per sortable column: its label, its URL, and its current state.

    Clicking the active column flips the direction; clicking any other column
    starts it descending, which is the useful end of every one of these scales.
    Paging resets, because page 40 of one ordering is not page 40 of another.
    """
    links: list[dict[str, object]] = []
    for column, label in SORT_LABELS.items():
        active = column is query.sort
        direction = (
            SortDirection.ASC
            if active and query.direction is SortDirection.DESC
            else SortDirection.DESC
        )
        links.append(
            {
                "column": column.value,
                "label": label,
                "active": active,
                "direction": query.direction.value if active else "",
                "url": _url(query, sort=column.value, direction=direction.value, page=1),
            }
        )
    return links


def _figure(points: tuple[ChartPoint, ...], page: SignalPage) -> dict[str, Any]:
    """The Plotly figure spec: IC025 against reported cases.

    Built as a plain dict and handed to plotly.js, so no Python plotting
    dependency is added for one chart.

    The caption is part of the figure rather than the surrounding page. A
    screenshot of a filtered chart travels without the page around it, and a
    thousand points out of two and a half million read as the whole corpus
    unless the image itself says otherwise. The cap and the filter are therefore
    in the title, where they cannot be cropped off separately from the data.
    """
    flagged = [point for point in points if point.flagged]
    plain = [point for point in points if not point.flagged]
    shown = len(points)

    subtitle = (
        f"{shown:,} of {page.total:,} matching pairs, those with the most cases "
        f"(cap {CHART_POINTS:,}). Filter: {page.query.describe_filter()}. "
        f"Run {page.run_id}."
    )

    return {
        "data": [
            _trace(flagged, "ROR, PRR and BCPNN all flag", _FLAGGED_COLOUR),
            _trace(plain, "Not flagged by all three", _UNFLAGGED_COLOUR),
        ],
        "layout": {
            "title": {
                "text": (
                    "IC025 against reported cases"
                    f'<br><span style="font-size:11px;color:#475569">{subtitle}</span>'
                ),
                "x": 0,
                "xanchor": "left",
                "font": {"size": 15},
            },
            "xaxis": {
                "title": {"text": "Co-reported cases (a), log scale"},
                "type": "log",
                "gridcolor": "#e2e8f0",
                "zeroline": False,
            },
            "yaxis": {
                "title": {"text": "IC025"},
                "gridcolor": "#e2e8f0",
                "zeroline": False,
            },
            # The BCPNN signal rule is IC025 > 0. Drawing it makes the boundary
            # between the two colours readable rather than asserted.
            "shapes": [
                {
                    "type": "line",
                    "xref": "paper",
                    "x0": 0,
                    "x1": 1,
                    "yref": "y",
                    "y0": 0,
                    "y1": 0,
                    "line": {"color": "#dc2626", "width": 1, "dash": "dot"},
                }
            ],
            "annotations": [
                {
                    "xref": "paper",
                    "yref": "y",
                    "x": 1,
                    "y": 0,
                    "xanchor": "right",
                    "yanchor": "bottom",
                    "text": "IC025 = 0, the BCPNN signal threshold",
                    "showarrow": False,
                    "font": {"size": 10, "color": "#dc2626"},
                }
            ],
            "margin": {"l": 60, "r": 20, "t": 70, "b": 55},
            "height": 380,
            "paper_bgcolor": "#ffffff",
            "plot_bgcolor": "#ffffff",
            "legend": {"orientation": "h", "y": -0.22, "x": 0},
            "hovermode": "closest",
        },
        "config": {"displayModeBar": False, "responsive": True},
    }


def _trace(points: list[ChartPoint], name: str, colour: str) -> dict[str, Any]:
    return {
        "type": "scatter",
        "mode": "markers",
        "name": name,
        "x": [point.a for point in points],
        "y": [round(point.ic025, 4) for point in points],
        "text": [f"{point.drug} / {point.pt}" for point in points],
        "hovertemplate": "%{text}<br>cases %{x:,}<br>IC025 %{y:.2f}<extra></extra>",
        "marker": {"color": colour, "size": 6, "opacity": 0.75},
    }
