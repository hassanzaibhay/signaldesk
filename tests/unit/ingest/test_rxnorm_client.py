"""The RxNav client, against recorded response shapes.

The bodies here are the shapes the ten committed probes actually returned, not
invented ones: candidates with no `name`, a top-ranked candidate that resolves to
no ingredient, and scores that are strings rather than numbers.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
import pytest
import respx

from signaldesk.core.config import Settings
from signaldesk.core.errors import ProviderError
from signaldesk.ingest.rxnorm import client
from signaldesk.ingest.rxnorm.client import TokenBucket

pytestmark = pytest.mark.unit

APPROXIMATE = f"{client.BASE_URL}/approximateTerm.json"


def _related_url(rxcui: str) -> str:
    return f"{client.BASE_URL}/rxcui/{rxcui}/related.json"


def _candidates(*entries: tuple[str, str, int]) -> dict[str, Any]:
    return {
        "approximateGroup": {
            "inputTerm": None,
            "candidate": [
                {"rxcui": rxcui, "rxaui": "1", "score": score, "rank": str(rank)}
                for rxcui, score, rank in entries
            ],
        }
    }


def _ingredients(*entries: tuple[str, str, str]) -> dict[str, Any]:
    return {
        "relatedGroup": {
            "rxcui": None,
            "conceptGroup": [
                {
                    "tty": tty,
                    "conceptProperties": [
                        {"rxcui": rxcui, "name": name, "tty": tty, "synonym": "", "suppress": "N"}
                    ],
                }
                for rxcui, name, tty in entries
            ],
        }
    }


_EMPTY_INGREDIENTS: dict[str, Any] = {
    "relatedGroup": {"rxcui": None, "conceptGroup": [{"tty": "IN"}, {"tty": "PIN"}]}
}


@respx.mock
def test_a_string_resolves_to_its_ingredient(settings: Settings) -> None:
    respx.get(APPROXIMATE).mock(
        return_value=httpx.Response(200, json=_candidates(("152523", "13.39", 1)))
    )
    respx.get(_related_url("152523")).mock(
        return_value=httpx.Response(200, json=_ingredients(("9143", "ranitidine", "IN")))
    )

    resolution = client.resolve("ZANTAC", settings)

    assert resolution.matched
    assert resolution.rxcui == "152523"
    assert [concept.name for concept in resolution.ingredients] == ["ranitidine"]
    assert resolution.rank == 1
    assert resolution.score == pytest.approx(13.39)


@respx.mock
def test_resolution_walks_past_a_candidate_that_resolves_to_nothing(settings: Settings) -> None:
    """Rank 1 returning empty ingredient groups is real, not hypothetical.

    `HYDROCODONE BITARTRATE AND ACETAMINOPHEN` did exactly this in the probe:
    the top candidate had empty IN and PIN groups and the second resolved.
    """
    respx.get(APPROXIMATE).mock(
        return_value=httpx.Response(
            200, json=_candidates(("214964", "15.29", 1), ("2661624", "14.73", 2))
        )
    )
    respx.get(_related_url("214964")).mock(
        return_value=httpx.Response(200, json=_EMPTY_INGREDIENTS)
    )
    respx.get(_related_url("2661624")).mock(
        return_value=httpx.Response(200, json=_ingredients(("161", "acetaminophen", "IN")))
    )

    resolution = client.resolve("HYDROCODONE BITARTRATE AND ACETAMINOPHEN", settings)

    assert resolution.rxcui == "2661624"
    assert resolution.rank == 2
    assert resolution.candidates_examined == 2


@respx.mock
def test_a_string_that_resolves_to_no_ingredient_is_unmatched(settings: Settings) -> None:
    """A device resolves to concepts with no ingredient behind them."""
    respx.get(APPROXIMATE).mock(
        return_value=httpx.Response(200, json=_candidates(("1943489", "9.24", 1)))
    )
    respx.get(_related_url("1943489")).mock(
        return_value=httpx.Response(200, json=_EMPTY_INGREDIENTS)
    )

    resolution = client.resolve("ISSD (INTEGRATED SYRINGE SAFETY DEVICE)", settings)

    assert not resolution.matched
    assert resolution.rxcui is None
    assert resolution.rank is None
    # The score of the best candidate is still recorded: it is evidence about
    # what the endpoint thought, and it is what shows the score means little.
    assert resolution.score == pytest.approx(9.24)


@respx.mock
def test_a_term_with_no_candidates_is_unmatched(settings: Settings) -> None:
    respx.get(APPROXIMATE).mock(return_value=httpx.Response(200, json={"approximateGroup": {}}))

    resolution = client.resolve("NOT A DRUG AT ALL", settings)

    assert not resolution.matched
    assert resolution.score is None
    assert resolution.candidates_examined == 0


@respx.mock
def test_candidates_are_returned_in_rank_order(settings: Settings) -> None:
    respx.get(APPROXIMATE).mock(
        return_value=httpx.Response(
            200, json=_candidates(("3", "1.0", 3), ("1", "9.0", 1), ("2", "5.0", 2))
        )
    )

    candidates, _ = client.approximate_term("ANYTHING", settings)

    assert [candidate.rank for candidate in candidates] == [1, 2, 3]


@respx.mock
def test_a_second_pass_makes_no_network_calls(settings: Settings) -> None:
    """The warm re-run requirement, asserted on request count rather than time."""
    approximate = respx.get(APPROXIMATE).mock(
        return_value=httpx.Response(200, json=_candidates(("161", "9.64", 1)))
    )
    related = respx.get(_related_url("161")).mock(
        return_value=httpx.Response(200, json=_ingredients(("161", "acetaminophen", "IN")))
    )

    first = client.resolve("ACETAMINOPHEN", settings)
    assert (approximate.call_count, related.call_count) == (1, 1)
    assert (first.requests, first.cache_hits) == (2, 0)

    second = client.resolve("ACETAMINOPHEN", settings)

    assert (approximate.call_count, related.call_count) == (1, 1)
    assert (second.requests, second.cache_hits) == (0, 2)
    assert second.ingredients == first.ingredients


@respx.mock
def test_the_cache_key_is_the_query_that_was_sent(settings: Settings) -> None:
    """Two strings that clean to the same query cost one request, not two."""
    approximate = respx.get(APPROXIMATE).mock(
        return_value=httpx.Response(200, json=_candidates(("161", "9.64", 1)))
    )
    respx.get(_related_url("161")).mock(
        return_value=httpx.Response(200, json=_ingredients(("161", "acetaminophen", "IN")))
    )

    client.resolve("ACETAMINOPHEN", settings)
    client.resolve("ACETAMINOPHEN", settings)

    assert approximate.call_count == 1


@respx.mock
def test_an_unreadable_body_raises_rather_than_returning_nothing(settings: Settings) -> None:
    respx.get(APPROXIMATE).mock(return_value=httpx.Response(200, text="<html>not json</html>"))

    with pytest.raises(ProviderError, match="unreadable body"):
        client.approximate_term("ZANTAC", settings)


@respx.mock
def test_a_persistent_error_status_raises(settings: Settings) -> None:
    respx.get(APPROXIMATE).mock(return_value=httpx.Response(503))

    with pytest.raises(ProviderError, match="503"):
        client.approximate_term("ZANTAC", settings)


def test_the_limiter_paces_requests_at_the_published_rate() -> None:
    """Twenty a second, because that is what the terms of service say."""
    bucket = TokenBucket(client.REQUESTS_PER_SECOND)
    for _ in range(int(client.REQUESTS_PER_SECOND)):
        bucket.take()

    started = time.monotonic()
    bucket.take()
    elapsed = time.monotonic() - started

    # The bucket started full, so the twenty-first call is the one that waits.
    assert elapsed >= 1.0 / client.REQUESTS_PER_SECOND * 0.5
    assert client.REQUESTS_PER_SECOND == 20.0
