"""Resolution priority, overrides, and combinations.

The rung order is the safety mechanism, so each rung is tested beating every rung
below it rather than only tested in isolation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from signaldesk.core.config import Settings
from signaldesk.core.errors import NormalizationError
from signaldesk.ingest.rxnorm import client, match

pytestmark = pytest.mark.unit

APPROXIMATE = f"{client.BASE_URL}/approximateTerm.json"
EXACT = f"{client.BASE_URL}/rxcui.json"


def _related_url(rxcui: str) -> str:
    return f"{client.BASE_URL}/rxcui/{rxcui}/related.json"


def _ingredient(rxcui: str, name: str) -> dict[str, Any]:
    return {
        "relatedGroup": {
            "conceptGroup": [
                {
                    "tty": "IN",
                    "conceptProperties": [{"rxcui": rxcui, "name": name, "tty": "IN"}],
                }
            ]
        }
    }


def _exact_hit(rxcui: str) -> dict[str, Any]:
    return {"idGroup": {"rxnormId": [rxcui]}}


_EXACT_MISS: dict[str, Any] = {"idGroup": {}}


def _write_overrides(path: Path, rows: str) -> Path:
    path.write_text(
        "raw_string,disposition,ingredient_rxcui,ingredient_name,note\n" + rows, encoding="utf-8"
    )
    return path


@respx.mock
def test_an_exact_term_never_reaches_the_approximate_endpoint(settings: Settings) -> None:
    """The rung order is the point: approximate matching is not consulted."""
    respx.get(EXACT).mock(return_value=httpx.Response(200, json=_exact_hit("6851")))
    respx.get(_related_url("6851")).mock(
        return_value=httpx.Response(200, json=_ingredient("6851", "methotrexate"))
    )
    approximate = respx.get(APPROXIMATE).mock(return_value=httpx.Response(200, json={}))

    result = match.match_string("prod_ai", "METHOTREXATE", settings=settings)

    assert result.method == match.METHOD_EXACT
    assert [concept.name for concept in result.ingredients] == ["methotrexate"]
    assert approximate.call_count == 0


@respx.mock
def test_approximate_matching_is_the_fallback_and_is_labelled(settings: Settings) -> None:
    respx.get(EXACT).mock(return_value=httpx.Response(200, json=_EXACT_MISS))
    respx.get(APPROXIMATE).mock(
        return_value=httpx.Response(
            200,
            json={
                "approximateGroup": {"candidate": [{"rxcui": "1191", "score": "3.31", "rank": "1"}]}
            },
        )
    )
    respx.get(_related_url("1191")).mock(
        return_value=httpx.Response(200, json=_ingredient("1191", "aspirin"))
    )

    result = match.match_string("drugname", "ACETYLSALICYLSAEURE", settings=settings)

    assert result.method == match.METHOD_APPROXIMATE
    assert result.score == pytest.approx(3.31)
    assert result.rank == 1


@respx.mock
def test_an_override_beats_both_endpoints(settings: Settings, tmp_path: Path) -> None:
    exact = respx.get(EXACT).mock(return_value=httpx.Response(200, json=_exact_hit("999")))
    approximate = respx.get(APPROXIMATE).mock(return_value=httpx.Response(200, json={}))
    overrides = match.load_overrides(
        _write_overrides(tmp_path / "o.csv", "MYSTERY PILL,ingredient,161,acetaminophen,curated\n")
    )

    result = match.match_string("drugname", "mystery pill", overrides=overrides, settings=settings)

    assert result.method == match.METHOD_OVERRIDE
    assert result.rxcui == "161"
    assert [concept.name for concept in result.ingredients] == ["acetaminophen"]
    assert exact.call_count == 0
    assert approximate.call_count == 0


@respx.mock
def test_a_string_ruled_not_a_drug_resolves_to_nothing_and_is_not_unmatched(
    settings: Settings, tmp_path: Path
) -> None:
    """`UNSPECIFIED INGREDIENT` resolves to caviar preparation if asked.

    So the override says it is not a drug, and that is a different outcome from
    failing to match: unmatched falls back into the comparison set as a raw
    string, and this must not.
    """
    exact = respx.get(EXACT).mock(return_value=httpx.Response(200, json=_exact_hit("1425214")))
    overrides = match.load_overrides(
        _write_overrides(
            tmp_path / "o.csv",
            "UNSPECIFIED INGREDIENT,not_a_drug,,,names no substance at all\n",
        )
    )

    result = match.match_string(
        "prod_ai", "UNSPECIFIED INGREDIENT", overrides=overrides, settings=settings
    )

    assert result.method == match.METHOD_OVERRIDE_NOT_A_DRUG
    assert result.ingredients == ()
    assert not result.matched
    assert exact.call_count == 0


@respx.mock
def test_a_combination_records_every_ingredient(settings: Settings) -> None:
    respx.get(EXACT, params={"name": "HYDROCODONE BITARTRATE"}).mock(
        return_value=httpx.Response(200, json=_exact_hit("5489"))
    )
    respx.get(EXACT, params={"name": "ACETAMINOPHEN"}).mock(
        return_value=httpx.Response(200, json=_exact_hit("161"))
    )
    respx.get(_related_url("5489")).mock(
        return_value=httpx.Response(200, json=_ingredient("5489", "hydrocodone"))
    )
    respx.get(_related_url("161")).mock(
        return_value=httpx.Response(200, json=_ingredient("161", "acetaminophen"))
    )

    result = match.match_string(
        "drugname", "HYDROCODONE BITARTRATE AND ACETAMINOPHEN", settings=settings
    )

    assert sorted(concept.name for concept in result.ingredients) == [
        "acetaminophen",
        "hydrocodone",
    ]
    assert result.components_total == 2
    assert result.components_matched == 2
    assert not result.is_partial
    # No single concept is the combination, so none is recorded as one.
    assert result.rxcui is None


@respx.mock
def test_a_partly_resolved_combination_records_the_shortfall(settings: Settings) -> None:
    """Half a combination is a shortened ingredient set, and it says so."""
    respx.get(EXACT, params={"name": "HYDROCODONE BITARTRATE"}).mock(
        return_value=httpx.Response(200, json=_exact_hit("5489"))
    )
    respx.get(EXACT, params={"name": "ACETAMINOPHEN"}).mock(
        return_value=httpx.Response(200, json=_EXACT_MISS)
    )
    respx.get(_related_url("5489")).mock(
        return_value=httpx.Response(200, json=_ingredient("5489", "hydrocodone"))
    )
    respx.get(APPROXIMATE).mock(
        return_value=httpx.Response(200, json={"approximateGroup": {"candidate": []}})
    )

    result = match.match_string(
        "drugname", "HYDROCODONE BITARTRATE AND ACETAMINOPHEN", settings=settings
    )

    assert result.components_total == 2
    assert result.components_matched == 1
    assert result.is_partial
    assert len(result.ingredients) == 1


@respx.mock
def test_a_string_nothing_resolves_is_unmatched_not_guessed(settings: Settings) -> None:
    respx.get(EXACT).mock(return_value=httpx.Response(200, json=_EXACT_MISS))
    respx.get(APPROXIMATE).mock(
        return_value=httpx.Response(200, json={"approximateGroup": {"candidate": []}})
    )

    result = match.match_string("drugname", "QQQQ ZZZZ", settings=settings)

    assert result.method == match.METHOD_UNMATCHED
    assert result.ingredients == ()
    assert not result.matched


@respx.mock
def test_the_cleaned_string_is_what_gets_queried(settings: Settings) -> None:
    route = respx.get(EXACT).mock(return_value=httpx.Response(200, json=_exact_hit("82063")))
    respx.get(_related_url("82063")).mock(
        return_value=httpx.Response(200, json=_ingredient("7804", "oxycodone"))
    )

    result = match.match_string(
        "drugname", "OXYCODONE HCL TABLETS (RHODES 91?490)", settings=settings
    )

    assert route.calls.last.request.url.params["name"] == "OXYCODONE HCL"
    assert result.cleaned == "OXYCODONE HCL"
    assert result.folded == "OXYCODONE HCL TABLETS (RHODES 91?490)"
    assert "uninformative_brackets" in result.steps


def test_an_override_row_without_an_rxcui_raises(tmp_path: Path) -> None:
    path = _write_overrides(tmp_path / "o.csv", "SOMETHING,ingredient,,,no id given\n")

    with pytest.raises(NormalizationError, match="needs an rxcui"):
        match.load_overrides(path)


def test_a_not_a_drug_row_without_a_reason_raises(tmp_path: Path) -> None:
    path = _write_overrides(tmp_path / "o.csv", "SOMETHING,not_a_drug,,,\n")

    with pytest.raises(NormalizationError, match="needs a note"):
        match.load_overrides(path)


def test_a_not_a_drug_row_carrying_an_rxcui_raises(tmp_path: Path) -> None:
    path = _write_overrides(tmp_path / "o.csv", "SOMETHING,not_a_drug,161,acetaminophen,why\n")

    with pytest.raises(NormalizationError, match="cannot carry an rxcui"):
        match.load_overrides(path)


def test_an_unknown_disposition_raises(tmp_path: Path) -> None:
    path = _write_overrides(tmp_path / "o.csv", "SOMETHING,maybe,161,acetaminophen,unsure\n")

    with pytest.raises(NormalizationError, match="disposition must be"):
        match.load_overrides(path)


def test_a_missing_override_file_is_not_an_error(tmp_path: Path) -> None:
    assert match.load_overrides(tmp_path / "absent.csv") == {}
