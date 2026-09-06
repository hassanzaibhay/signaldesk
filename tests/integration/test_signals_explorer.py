"""The signals explorer end to end, over a fixture run and real label rows.

The corpus is redirected to a temporary directory rather than read from the
container's volume, so these assertions hold on a machine that has never built
FAERS. The label tables are real Postgres rows, because the three-state
resolution is the part of this page most able to look right while being wrong.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import polars as pl
import pytest
from django.test import Client
from django.urls import reverse

from signaldesk.analytics.signals import signal_root
from signaldesk.core.config import get_settings
from signaldesk.ingest.spl.manifest import SOURCE as SPL_SOURCE
from signaldesk.web.documents.models import LabelDocument, LabelDrugKey, LabelSection
from signaldesk.web.signals.labels import LabelState, statuses_for
from signaldesk.web.signals.models import IngestManifest
from signaldesk.web.signals.query import PINNED_RUN_ID

pytestmark = [pytest.mark.integration, pytest.mark.django_db(transaction=True)]

#: Three drugs, one per label state.
LABELLED = "EXAMPLAMAB"
QUERIED = "NOLABELIUM"
UNSCOPED = "NEVERASKEDINE"

_ROWS = {
    "drug": [LABELLED, QUERIED, UNSCOPED, LABELLED, "BELOWFLOORAZOLE"],
    "pt": ["RASH", "RASH", "NAUSEA", "PYREXIA", "RASH"],
    "a": [900, 40, 120, 7, 2],
    "ror": [12.5, 3.25, 6.0, None, 99.0],
    "ror_lower": [11.0, 2.9, 5.4, None, 50.0],
    "ror_upper": [14.0, 3.6, 6.7, None, 150.0],
    "prr": [10.0, 3.0, 5.5, None, 80.0],
    "prr_lower": [9.1, 2.7, 5.0, None, 40.0],
    "prr_upper": [11.0, 3.3, 6.0, None, 120.0],
    "ic025": [3.4, 1.2, 2.1, None, 4.9],
    "corrected": [False, False, False, True, False],
    "flag_ror": [True, True, True, False, True],
    "flag_prr": [True, True, True, False, True],
    "flag_bcpnn": [True, True, True, False, True],
    "flag_ror_prr_bcpnn": [True, True, True, False, True],
}


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point the whole process at a temporary data directory holding one run.

    The environment is redirected and the settings cache cleared, rather than
    patching each module's imported name: ``signal_root`` and the connection
    factory both reach for the process settings, and patching one of them would
    leave the page reading half a configuration.
    """
    data_dir = tmp_path / "data"
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    get_settings.cache_clear()
    partition = signal_root() / f"run={PINNED_RUN_ID}"
    partition.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(_ROWS).write_parquet(partition / "part-0.parquet")
    yield data_dir
    get_settings.cache_clear()


@pytest.fixture
def label_rows() -> None:
    """One drug string with a label, and one that was queried and had none."""
    document = LabelDocument.objects.create(
        set_id="8f2c1a44-0000-4a11-9f00-00000000abcd",
        brand_names=["EXAMPLAMAB"],
    )
    LabelSection.objects.create(
        document=document, section_code="boxed_warning", ordinal=0, text="WARNING: rash."
    )
    LabelSection.objects.create(
        document=document, section_code="adverse_reactions", ordinal=0, text="Rash."
    )
    LabelDrugKey.objects.create(
        folded_string=LABELLED, query=LABELLED, route="cleaned_string", document=document
    )
    IngestManifest.objects.create(
        source=SPL_SOURCE,
        unit=_unit(QUERIED),
        status=IngestManifest.Status.COMPLETED,
        row_counts={"documents": 0, "sections": 0},
    )


def _unit(folded_string: str) -> str:
    from signaldesk.web.signals.labels import _manifest_unit

    return _manifest_unit(folded_string)


def _get(client: Client, query: str = "", **headers: str) -> str:
    response = client.get(reverse("signals:explorer") + query, **headers)
    assert response.status_code == 200
    return response.content.decode()


def test_the_page_renders_the_run_it_is_serving(client: Client, corpus: Path) -> None:
    body = _get(client)

    assert PINNED_RUN_ID in body
    assert "Drug-event signals" in body


def test_the_root_redirects_to_the_explorer(client: Client) -> None:
    response = client.get("/")

    assert response.status_code == 302
    assert response["Location"] == reverse("signals:explorer")


def test_pairs_below_the_minimum_cell_count_never_reach_the_page(
    client: Client, corpus: Path
) -> None:
    body = _get(client)

    assert "BELOWFLOORAZOLE" not in body


def test_an_htmx_request_returns_the_fragment_and_not_the_chrome(
    client: Client, corpus: Path
) -> None:
    """The swap target is the results block; sending the whole page would nest it."""
    fragment = _get(client, HTTP_HX_REQUEST="true")

    assert "<html" not in fragment
    assert "Drug-event signals" not in fragment
    assert LABELLED in fragment


def test_the_filter_is_applied_server_side(client: Client, corpus: Path) -> None:
    fragment = _get(client, "?drug=nolabelium", HTTP_HX_REQUEST="true")

    assert QUERIED in fragment
    assert UNSCOPED not in fragment


def test_a_filter_matching_nothing_explains_why_rather_than_showing_an_empty_table(
    client: Client, corpus: Path
) -> None:
    """The single most useful line on the page for a reader new to this data."""
    body = _get(client, "?drug=NOTHINGLIKETHIS")

    assert "Nothing matches" in body
    assert "raw FAERS drug strings folded to upper case" in body
    assert "not over" in body and "RxNorm ingredients" in body
    assert "<tbody" not in body


def test_a_sort_column_outside_the_allowlist_falls_back_to_the_default(
    client: Client, corpus: Path
) -> None:
    body = _get(client, "?sort=drug%3B+DROP+TABLE+signal")

    assert "Cases (desc)" in body


def test_sorting_is_server_side_and_changes_the_order(client: Client, corpus: Path) -> None:
    """NOLABELIUM has 40 cases and NEVERASKEDINE 120, so the two swap places."""
    ascending = _get(client, "?sort=a&direction=asc", HTTP_HX_REQUEST="true")
    descending = _get(client, "?sort=a&direction=desc", HTTP_HX_REQUEST="true")

    assert ascending.index(QUERIED) < ascending.index(UNSCOPED)
    assert descending.index(UNSCOPED) < descending.index(QUERIED)


def test_the_mgps_column_is_present_and_holds_no_number(client: Client, corpus: Path) -> None:
    """The column exists, every cell says why it is empty, and none of them is a value."""
    body = _get(client)
    served = [value for value in _ROWS["a"] if value >= 3]
    cells = re.findall(r"<td[^>]*>\s*<span[^>]*>\s*not quotable\s*</span>\s*</td>", body)

    assert re.search(r"<th[^>]*>\s*MGPS\s*</th>", body) is not None
    assert len(cells) == len(served)
    assert "MGPS is not quotable for this run." in body
    # EBGM and EBGM05 are named only where they are being explained - the banner
    # and the cell tooltip. Stripping the tooltips leaves the text a reader
    # actually sees in the table, and neither quantity may appear there.
    table = body[body.index("<tbody") : body.index("</tbody>")]
    visible = re.sub(r'title="[^"]*"', "", table).lower()
    assert "ebgm" not in visible


def test_each_label_state_says_something_rather_than_nothing(
    client: Client, corpus: Path, label_rows: None
) -> None:
    body = _get(client)

    assert "Label held" in body
    assert "Queried, no label" in body
    assert "Not in label scope" in body


def test_the_label_states_resolve_from_the_stored_rows(label_rows: None) -> None:
    statuses = statuses_for([LABELLED, QUERIED, UNSCOPED])

    assert statuses[LABELLED].state is LabelState.LABELLED
    assert statuses[LABELLED].documents == 1
    assert statuses[LABELLED].sections == ("Boxed warning", "Adverse reactions")
    assert statuses[QUERIED].state is LabelState.QUERIED_NO_LABEL
    assert statuses[UNSCOPED].state is LabelState.OUT_OF_SCOPE


def test_a_string_with_no_label_row_is_still_given_a_state(label_rows: None) -> None:
    """Every drug on a page gets a cell with words in it, whatever is known."""
    statuses = statuses_for([LABELLED, QUERIED, UNSCOPED])

    assert set(statuses) == {LABELLED, QUERIED, UNSCOPED}
    assert all(status.display.strip() for status in statuses.values())


def test_the_page_names_the_committed_artifact_behind_its_label_figures(
    client: Client, corpus: Path
) -> None:
    body = _get(client)

    assert "spl_ingest_" in body
    assert "distinct queries resolved" in body
    assert "selected slots" in body


def test_a_run_with_no_committed_artifact_says_its_numbers_are_not_published(
    client: Client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fallback is visible on the page, never silent."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "other"))
    get_settings.cache_clear()
    partition = signal_root() / "run=20200101T000000Z"
    partition.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(_ROWS).write_parquet(partition / "part-0.parquet")
    try:
        body = _get(client)
    finally:
        get_settings.cache_clear()

    assert "has no record under" in body
    assert "is a published figure until that run is collected" in body
