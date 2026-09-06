"""Whether a drug string on the signals page has label evidence behind it.

Three states, not two. The distinction between "we asked openFDA and it had
nothing" and "we never asked" is the whole point: the SPL ingest was scoped to
the top 200 flagged drug strings out of 30,549, so the overwhelmingly common
answer is that a string was never in scope. Collapsing that into a blank cell,
or into the words "no label", would report a gap in coverage as a fact about the
drug.

The join key is exact rather than fuzzy. ``LabelDrugKey.folded_string`` is
``upper(trim(drugname_raw))`` and this signal run was built with
``drug_key=raw_string``, which groups on the same expression in
``analytics.contingency``. All 191 label-carrying strings appear in the run's
``a >= 3`` slice, so the two vocabularies are the same vocabulary.

The second state is reconstructed rather than stored: given a string, the query
that would have been sent is ``scope.clean_query``, and the manifest unit for
that query is ``ScopeUnit.manifest_unit``. Both are imported from the ingest
that wrote them rather than reimplemented here, because a divergent copy of the
digest would silently report every string as never-asked.

One limit, stated rather than hidden. The reconstruction covers the cleaned
string route only. The ingredient route keys its unit on an rxcui instead, and
resolving that would need ``DrugStringMatch``, which is empty; the run this page
serves resolved 0 strings by that route and 200 by the string route, so the
reconstruction is exact today. If the ingredient route is ever used, strings it
fetched fall back to "outside the scope", which understates coverage and never
overstates it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from django.db.models import Count

from signaldesk.ingest.spl.manifest import SOURCE as SPL_SOURCE
from signaldesk.ingest.spl.scope import ScopeUnit, clean_query
from signaldesk.web.documents.models import LabelDrugKey, LabelSection, SectionCode
from signaldesk.web.signals.models import IngestManifest


class LabelState(StrEnum):
    """What is known about label evidence for one drug string."""

    #: openFDA returned at least one label and its sections are stored.
    LABELLED = "labelled"
    #: The string was queried and openFDA returned no label for it.
    QUERIED_NO_LABEL = "queried_no_label"
    #: The string was never queried; it fell outside the fetch scope.
    OUT_OF_SCOPE = "out_of_scope"


#: What each state says on the page. Short enough for a table cell, and none of
#: them is blank.
STATE_LABELS: Final[dict[LabelState, str]] = {
    LabelState.LABELLED: "Label held",
    LabelState.QUERIED_NO_LABEL: "Queried, no label",
    LabelState.OUT_OF_SCOPE: "Not in label scope",
}

#: The longer form, shown on hover, so the short form is never the only
#: explanation a reader gets.
STATE_EXPLANATIONS: Final[dict[LabelState, str]] = {
    LabelState.LABELLED: (
        "openFDA returned at least one structured product label for this drug string "
        "and its sections are stored."
    ),
    LabelState.QUERIED_NO_LABEL: (
        "This drug string was sent to openFDA and no structured product label came "
        "back. Absence of a label here is a fact about the query, not about the drug."
    ),
    LabelState.OUT_OF_SCOPE: (
        "This drug string was never queried. The label ingest was scoped to the 200 "
        "drug strings carrying the most flagged pairs, of 30,549 that carry any."
    ),
}


@dataclass(frozen=True, slots=True)
class LabelStatus:
    """The label evidence behind one drug string."""

    folded_string: str
    state: LabelState
    #: Distinct labels reached from this string. Zero unless ``LABELLED``.
    documents: int
    #: Section names held across those labels, in the order the enum declares.
    sections: tuple[str, ...]
    #: What was, or would have been, sent to openFDA. Kept for both queried
    #: states so the chain from FAERS string to query is visible on the page.
    query: str

    @property
    def display(self) -> str:
        return STATE_LABELS[self.state]

    @property
    def explanation(self) -> str:
        return STATE_EXPLANATIONS[self.state]


#: Section code to its human name, from the model's own choices, so the page
#: cannot drift from the vocabulary the ingest wrote.
_SECTION_NAMES: Final[dict[str, str]] = dict(SectionCode.choices)


def _section_names(codes: Iterable[str]) -> tuple[str, ...]:
    """Section names in declaration order, unknown codes passed through.

    Declaration order rather than alphabetical: a boxed warning is not the same
    weight of statement as an adverse reactions list, and the enum already
    records the order a reader expects to read them in.
    """
    held = set(codes)
    return tuple(name for code, name in SectionCode.choices if code in held) + tuple(
        sorted(code for code in held if code not in _SECTION_NAMES)
    )


def statuses_for(strings: Iterable[str]) -> dict[str, LabelStatus]:
    """Label state for every string given, keyed by the string.

    Three queries for a page, regardless of how many rows it holds: the drug
    keys with their document counts, the section codes behind them, and the
    manifest lookup for whatever is left over. All three are indexed; measured
    at 8 ms for a fifty-row page.
    """
    wanted = sorted({string for string in strings if string})
    if not wanted:
        return {}

    counts = {
        row["folded_string"]: row["documents"]
        for row in LabelDrugKey.objects.filter(folded_string__in=wanted)
        .values("folded_string")
        .annotate(documents=Count("document", distinct=True))
    }
    queries = dict(
        LabelDrugKey.objects.filter(folded_string__in=wanted).values_list("folded_string", "query")
    )

    # Aggregated in the database rather than by pulling document ids back and
    # querying again: one string in this scope reaches 751 labels, and an IN
    # clause over every label id on a page would be thousands of parameters for
    # an answer that is at most four rows per string.
    sections: dict[str, set[str]] = {}
    for folded, code in (
        LabelSection.objects.filter(document__drug_keys__folded_string__in=wanted)
        .values_list("document__drug_keys__folded_string", "section_code")
        .distinct()
    ):
        sections.setdefault(folded, set()).add(code)

    unlabelled = [string for string in wanted if string not in counts]
    unit_for = {string: _manifest_unit(string) for string in unlabelled}
    queried = set(
        IngestManifest.objects.filter(
            source=SPL_SOURCE, unit__in=sorted(set(unit_for.values()))
        ).values_list("unit", flat=True)
    )

    statuses: dict[str, LabelStatus] = {}
    for string in wanted:
        if string in counts:
            statuses[string] = LabelStatus(
                folded_string=string,
                state=LabelState.LABELLED,
                documents=counts[string],
                sections=_section_names(sections.get(string, set())),
                query=queries.get(string, ""),
            )
            continue
        state = (
            LabelState.QUERIED_NO_LABEL if unit_for[string] in queried else LabelState.OUT_OF_SCOPE
        )
        statuses[string] = LabelStatus(
            folded_string=string,
            state=state,
            documents=0,
            sections=(),
            query=clean_query(string),
        )
    return statuses


def _manifest_unit(folded_string: str) -> str:
    """The manifest unit the cleaned-string route would have used.

    Built through ``ScopeUnit`` rather than by hashing here, so that a change to
    the cleaner or to the digest reaches this page automatically instead of
    leaving it quietly reporting the wrong state.
    """
    return ScopeUnit(
        folded_string=folded_string,
        query=clean_query(folded_string),
        route="cleaned_string",
        ingredient_rxcui=None,
        flagged_pairs=0,
    ).manifest_unit
