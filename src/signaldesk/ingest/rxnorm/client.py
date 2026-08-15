"""The RxNav client.

Two calls make a match: `approximateTerm` proposes concepts for a string, and
`rxcui/{id}/related` resolves a concept to its ingredients. Both go through the
shared HTTP client, so both are cached on disk and a second pass over the same
strings costs no requests at all.

## The published limit, and why nothing here tries to beat it

NLM's terms of service state, verbatim:

> In order to avoid overloading the RxNav servers, NLM requires that users of the
> APIs (RxNorm, RxTerms, Prescribable RxNorm, and RxClass) send no more than 20
> requests per second per IP address.

https://lhncbc.nlm.nih.gov/RxNav/TermsofService.html

The limiter below is process-global rather than per client instance, because the
limit is per IP: term matching, candidate walking and any class lookup all draw
on one budget, and three components each politely staying under 20 would put 60
requests a second on the servers.

## What the ten committed probes established about the responses

Recorded in `evals/history/rxnav_probe_20260814T060235Z.json` before this file
existed, because both findings change the shape of the code:

* **The score is not a confidence.** It ranges from 3.31 to 42.40 across ten
  terms and tracks query length. It is recorded because it was returned, not
  because it means anything; see docs/adr/0008.
* **The top-ranked candidate can resolve to nothing.** For a combination string
  the rank 1 candidate returned empty `IN` and `PIN` groups while rank 2
  resolved correctly. Resolution therefore walks candidates in rank order and
  records which one produced the ingredients.

`name` is absent on candidates from several sources, so nothing here may depend
on it being present.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Final

import httpx
from pydantic import BaseModel, Field, ValidationError

from signaldesk.core.config import Settings, get_settings
from signaldesk.core.errors import ProviderError
from signaldesk.core.http import CACHE_HEADER, get, is_cached
from signaldesk.core.logging import get_logger

log = get_logger(__name__)

BASE_URL: Final[str] = "https://rxnav.nlm.nih.gov/REST"

#: The published ceiling. Not a tuning knob.
REQUESTS_PER_SECOND: Final[float] = 20.0

#: Candidates requested per term. Enough to walk past a top candidate that
#: resolves to no ingredient, few enough not to ask for pages nobody reads.
MAX_ENTRIES: Final[int] = 5

#: Ingredient term types. `IN` is the ingredient; `PIN` is its precise form, kept
#: because a salt is what the label names and the ingredient is what groups.
INGREDIENT_TTYS: Final[tuple[str, ...]] = ("IN", "PIN")


class TokenBucket:
    """A process-global rate limiter, since the published limit is per IP."""

    def __init__(self, per_second: float) -> None:
        self._per_second = per_second
        self._lock = threading.Lock()
        self._allowance = per_second
        self._checked_at = time.monotonic()

    def take(self) -> None:
        """Block until one request may be sent."""
        with self._lock:
            now = time.monotonic()
            self._allowance = min(
                self._per_second,
                self._allowance + (now - self._checked_at) * self._per_second,
            )
            self._checked_at = now
            if self._allowance < 1.0:
                delay = (1.0 - self._allowance) / self._per_second
                time.sleep(delay)
                self._checked_at = time.monotonic()
                self._allowance = 0.0
            else:
                self._allowance -= 1.0


_LIMITER = TokenBucket(REQUESTS_PER_SECOND)


class Candidate(BaseModel):
    """One concept RxNav proposes for a term."""

    rxcui: str
    rxaui: str | None = None
    score: float
    rank: int
    name: str | None = None
    source: str | None = None


class _ApproximateGroup(BaseModel):
    candidate: list[Candidate] = Field(default_factory=list)


class _ApproximateResponse(BaseModel):
    approximateGroup: _ApproximateGroup = Field(default_factory=_ApproximateGroup)  # noqa: N815


class Concept(BaseModel):
    """An ingredient concept as `related` returns it."""

    rxcui: str
    name: str
    tty: str


class _ConceptGroup(BaseModel):
    tty: str
    conceptProperties: list[Concept] = Field(default_factory=list)  # noqa: N815


class _RelatedGroup(BaseModel):
    conceptGroup: list[_ConceptGroup] = Field(default_factory=list)  # noqa: N815


class _RelatedResponse(BaseModel):
    relatedGroup: _RelatedGroup = Field(default_factory=_RelatedGroup)  # noqa: N815


@dataclass(frozen=True, slots=True)
class Resolution:
    """What one cleaned string resolved to, with the evidence for it."""

    query: str
    rxcui: str | None
    ingredients: tuple[Concept, ...]
    score: float | None
    rank: int | None
    candidates_examined: int
    requests: int
    cache_hits: int

    @property
    def matched(self) -> bool:
        return bool(self.ingredients)


def _fetch(url: str, params: dict[str, str], settings: Settings | None = None) -> httpx.Response:
    """One GET, rate limited only when it will actually leave the machine.

    Taking a token before checking the cache would throttle cache hits, and a
    warm pass over half a million lookups is all cache hits: it would sleep for
    seven hours to send nothing.
    """
    settings = settings or get_settings()
    if not is_cached(url, params=params, settings=settings):
        _LIMITER.take()
    response = get(url, params=params, settings=settings)
    if response.status_code != httpx.codes.OK:
        message = f"RxNav returned {response.status_code} for {url}"
        raise ProviderError(message)
    return response


def _from_cache(response: httpx.Response) -> bool:
    return bool(response.headers.get(CACHE_HEADER) == "hit")


def approximate_term(
    term: str, settings: Settings | None = None, *, max_entries: int = MAX_ENTRIES
) -> tuple[tuple[Candidate, ...], bool]:
    """Candidate concepts for a term, best rank first, and whether it was cached."""
    response = _fetch(
        f"{BASE_URL}/approximateTerm.json",
        {"term": term, "maxEntries": str(max_entries)},
        settings,
    )
    try:
        parsed = _ApproximateResponse.model_validate(response.json())
    except (ValidationError, ValueError) as error:
        message = f"RxNav approximateTerm returned an unreadable body for {term!r}"
        raise ProviderError(message) from error
    ordered = tuple(sorted(parsed.approximateGroup.candidate, key=lambda item: item.rank))
    return ordered, _from_cache(response)


def ingredients_for(
    rxcui: str, settings: Settings | None = None
) -> tuple[tuple[Concept, ...], bool]:
    """The ingredient concepts a concept resolves to, and whether it was cached.

    Empty is a real answer: a device, or a concept with no ingredient behind it,
    resolves to nothing and must not be coerced into something.
    """
    response = _fetch(
        f"{BASE_URL}/rxcui/{rxcui}/related.json", {"tty": " ".join(INGREDIENT_TTYS)}, settings
    )
    try:
        parsed = _RelatedResponse.model_validate(response.json())
    except (ValidationError, ValueError) as error:
        message = f"RxNav related returned an unreadable body for rxcui {rxcui}"
        raise ProviderError(message) from error
    concepts = tuple(
        concept
        for group in parsed.relatedGroup.conceptGroup
        if group.tty in INGREDIENT_TTYS
        for concept in group.conceptProperties
    )
    return concepts, _from_cache(response)


class _IdGroup(BaseModel):
    #: Absent entirely when there is no exact match, rather than empty. Read the
    #: absence, do not infer it from a length.
    rxnormId: list[str] | None = None  # noqa: N815


class _ExactResponse(BaseModel):
    idGroup: _IdGroup = Field(default_factory=_IdGroup)  # noqa: N815


def exact_term(term: str, settings: Settings | None = None) -> tuple[str | None, bool]:
    """The concept whose name is exactly this string, if there is one.

    Worth a request of its own before falling back to approximate matching,
    because the two endpoints behave differently on the input that matters:
    `UNSPECIFIED INGREDIENT` returns nothing here and `caviar preparation` from
    the approximate matcher. Exact matching refuses what it does not know, which
    is the property the rung order is built on.
    """
    response = _fetch(f"{BASE_URL}/rxcui.json", {"name": term, "search": "0"}, settings)
    try:
        parsed = _ExactResponse.model_validate(response.json())
    except (ValidationError, ValueError) as error:
        message = f"RxNav rxcui returned an unreadable body for {term!r}"
        raise ProviderError(message) from error
    identifiers = parsed.idGroup.rxnormId or []
    return (identifiers[0] if identifiers else None), _from_cache(response)


def resolve_exact(query: str, settings: Settings | None = None) -> Resolution | None:
    """Resolve a string by exact name, or None if it is not a term."""
    rxcui, cached = exact_term(query, settings)
    requests = 0 if cached else 1
    cache_hits = 1 if cached else 0
    if rxcui is None:
        return None

    concepts, hit = ingredients_for(rxcui, settings)
    requests += 0 if hit else 1
    cache_hits += 1 if hit else 0
    if not concepts:
        return None
    return Resolution(
        query=query,
        rxcui=rxcui,
        ingredients=concepts,
        score=None,
        rank=None,
        candidates_examined=1,
        requests=requests,
        cache_hits=cache_hits,
    )


def resolve(query: str, settings: Settings | None = None) -> Resolution:
    """Match one cleaned string to ingredient concepts.

    Walks candidates in rank order until one resolves to at least one ingredient,
    because the probe found rank 1 returning empty ingredient groups where rank 2
    resolved. The rank that succeeded is recorded: a match that needed the fourth
    candidate is weaker evidence than one that resolved at the first, and that is
    a signal the confidence mechanism may use (docs/adr/0008).
    """
    candidates, cached = approximate_term(query, settings)
    requests = 0 if cached else 1
    cache_hits = 1 if cached else 0

    examined = 0
    for candidate in candidates:
        examined += 1
        concepts, hit = ingredients_for(candidate.rxcui, settings)
        requests += 0 if hit else 1
        cache_hits += 1 if hit else 0
        if concepts:
            return Resolution(
                query=query,
                rxcui=candidate.rxcui,
                ingredients=concepts,
                score=candidate.score,
                rank=candidate.rank,
                candidates_examined=examined,
                requests=requests,
                cache_hits=cache_hits,
            )

    log.debug("rxnorm.unresolved", query=query, candidates=len(candidates))
    return Resolution(
        query=query,
        rxcui=None,
        ingredients=(),
        score=candidates[0].score if candidates else None,
        rank=None,
        candidates_examined=examined,
        requests=requests,
        cache_hits=cache_hits,
    )
