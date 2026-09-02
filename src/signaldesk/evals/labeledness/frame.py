"""The sampling frame: which drug strings and which documents are eligible.

Two exclusions, both settled in the P12 frame ruling and both stated here as
predicates rather than as counts, because a count is a description of one corpus
state and a predicate survives the corpus moving.

**No adverse-reactions text.** A document with no ``adverse_reactions`` section
carries nothing the labeledness question can be adjudicated against. This
removes documents whose only section is a warning, documents with several
sections none of which is adverse reactions, and documents with no sections at
all -- the last group being the one a "warnings-only" framing silently misses.

**Page cap.** openFDA paginates, the fetch stops at a cap, and a capped fetch
means the drug string's *document set* is truncated. The documents themselves
are complete. What is unsafe is the ``not-labelled`` verdict: the label that
describes the event may be one of the ones never fetched. The error is
asymmetric, so the exclusion is at the string level rather than the document
level, and a document reachable from some other string whose fetch was complete
stays in the frame.

``page_cap`` has three states and the third is the point. ``F4`` recorded the
flag on the manifest row so an attaching unit could carry it, and left rows
written before the flag existed reading ``null``. ``null`` means truncation was
never measured, which is not the same as measured-and-absent. This module
excludes ``null`` as ``UNKNOWN`` and refuses to coerce it: see
:func:`page_cap_state`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Final

from signaldesk.core.errors import AnnotationError
from signaldesk.core.logging import get_logger

log = get_logger(__name__)

#: The section this project adjudicates against. A document without it is out of
#: frame regardless of what else it carries.
PRIMARY_SECTION: Final = "adverse_reactions"

#: Read for the class-warning route, in the order the guideline searches them.
SUPPORTING_SECTIONS: Final[tuple[str, ...]] = (
    "boxed_warning",
    "warnings_and_cautions",
    "warnings",
)

SECTION_ORDER: Final[tuple[str, ...]] = (PRIMARY_SECTION, *SUPPORTING_SECTIONS)


class CapState(StrEnum):
    """Whether a fetch is known to have been truncated.

    Three members, and ``UNKNOWN`` is not a synonym for ``CLEAN``. Collapsing the
    two is exactly the defect this enum exists to make unrepresentable: a bool
    has no room for "never measured", and ``None`` coerced with ``bool()`` reads
    as ``False``, which is the reading that silently admits a truncated corpus.
    """

    CLEAN = "clean"
    CAPPED = "capped"
    UNKNOWN = "unknown"


def page_cap_state(value: object) -> CapState:
    """Map an artifact's ``page_cap`` field onto a :class:`CapState`.

    ``True`` and ``False`` are measurements a fetch made. ``None``, and a field
    that is absent entirely, are the absence of a measurement and become
    ``UNKNOWN``.

    Anything else is rejected rather than interpreted. ``0``, ``"false"`` and
    ``""`` are all falsey and all would read as "not capped" under a ``bool()``
    call; none of them is a page-cap measurement, and an artifact carrying one
    is malformed in a way that should stop the draw rather than quietly shrink
    the frame. ``bool`` is checked before ``int`` because ``bool`` is an ``int``
    and the order is the whole guard.
    """
    if value is None:
        return CapState.UNKNOWN
    if isinstance(value, bool):
        return CapState.CAPPED if value else CapState.CLEAN
    message = (
        f"page_cap must be true, false or null; got {value!r} of type "
        f"{type(value).__name__}. null means truncation was never measured and is "
        "excluded as unknown; a non-boolean is a malformed artifact and is not "
        "coerced."
    )
    raise AnnotationError(message)


@dataclass(frozen=True, slots=True)
class FetchUnit:
    """One openFDA query, and whether its fetch was truncated."""

    query: str
    folded_string: str
    state: CapState


def read_fetch_units(artifact: Path) -> list[FetchUnit]:
    """Read the per-unit page-cap states out of a committed SPL ingest artifact.

    The artifact rather than the live manifest table, deliberately. The artifact
    is committed, so the frame is reproducible from the repository; the manifest
    table is mutable and a later ingest would silently redefine a frame that has
    already been drawn from.
    """
    try:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        message = f"cannot read SPL ingest artifact {artifact}: {exc}"
        raise AnnotationError(message) from exc
    units = payload.get("units")
    if not isinstance(units, list):
        message = f"{artifact} has no units list; it is not an SPL ingest artifact"
        raise AnnotationError(message)

    parsed: list[FetchUnit] = []
    for index, row in enumerate(units):
        if not isinstance(row, Mapping):
            message = f"{artifact} units[{index}] is not an object"
            raise AnnotationError(message)
        folded = row.get("folded_string")
        query = row.get("query")
        if not isinstance(folded, str) or not isinstance(query, str):
            message = f"{artifact} units[{index}] has no folded_string/query pair"
            raise AnnotationError(message)
        # Absent and explicitly null are the same statement: nothing measured.
        parsed.append(
            FetchUnit(query=query, folded_string=folded, state=page_cap_state(row.get("page_cap")))
        )
    return parsed


def string_cap_states(units: Iterable[FetchUnit]) -> dict[str, CapState]:
    """Collapse the units reaching each drug string onto one state per string.

    A string reached by several queries is only clean when every one of them is.
    ``CAPPED`` outranks ``UNKNOWN`` in the reported reason because it is a
    measurement and ``UNKNOWN`` is the absence of one; both exclude, so the
    ordering affects the report and never the frame.
    """
    states: dict[str, CapState] = {}
    for unit in units:
        current = states.get(unit.folded_string)
        if current is None:
            states[unit.folded_string] = unit.state
            continue
        if current is CapState.CAPPED or unit.state is CapState.CAPPED:
            states[unit.folded_string] = CapState.CAPPED
        elif current is CapState.UNKNOWN or unit.state is CapState.UNKNOWN:
            states[unit.folded_string] = CapState.UNKNOWN
    return states


@dataclass(frozen=True, slots=True)
class DocumentRow:
    """One label document, reduced to what the frame decides on."""

    document_id: int
    set_id: str
    has_primary_section: bool


@dataclass(frozen=True, slots=True)
class ReachRow:
    """One (drug string, document) reach-through row."""

    folded_string: str
    query: str
    document_id: int


@dataclass(frozen=True, slots=True)
class FrameCounts:
    """Every count the frame is defined by, for the committed manifest.

    Recorded rather than recomputed at read time. A reviewer checking the
    published frame against the corpus needs the numbers the draw actually saw,
    not the numbers today's corpus would produce.
    """

    documents_total: int
    documents_with_primary_section: int
    documents_without_primary_section: int
    strings_total: int
    strings_clean: int
    strings_capped: int
    strings_unknown: int
    strings_eligible: int
    query_groups_eligible: int
    documents_eligible: int


@dataclass(frozen=True, slots=True)
class Frame:
    """The eligible population, and the counts that define it."""

    counts: FrameCounts
    #: Eligible drug string -> the eligible documents it reaches.
    documents_by_string: dict[str, tuple[int, ...]]
    #: Eligible drug string -> the openFDA query it was fetched under.
    query_by_string: dict[str, str]
    #: Query -> the eligible strings sharing it, ascending. Siblings for repeats.
    strings_by_query: dict[str, tuple[str, ...]]
    #: Eligible document id -> set_id.
    set_id_by_document: dict[int, str]
    excluded_strings: dict[str, CapState] = field(default_factory=dict)

    def documents_for_query(self, query: str) -> tuple[int, ...]:
        """Every eligible document reachable from any string in a query group."""
        reachable: set[int] = set()
        for name in self.strings_by_query.get(query, ()):
            reachable.update(self.documents_by_string.get(name, ()))
        return tuple(sorted(reachable))


def build_frame(
    *,
    documents: Sequence[DocumentRow],
    reach: Sequence[ReachRow],
    units: Sequence[FetchUnit],
) -> Frame:
    """Apply both exclusions and report what survives.

    Order matters only for the reporting. A string is eligible when its fetch is
    ``CLEAN`` and it reaches at least one document carrying adverse-reactions
    text; a string whose fetch was clean but whose every document lacks the
    section is out of frame with nothing to annotate rather than excluded by the
    page-cap rule, and the counts keep the two apart.
    """
    with_primary = {row.document_id for row in documents if row.has_primary_section}
    set_ids = {row.document_id: row.set_id for row in documents}

    states = string_cap_states(units)
    reach_by_string: dict[str, set[int]] = {}
    query_by_string: dict[str, str] = {}
    for row in reach:
        reach_by_string.setdefault(row.folded_string, set()).add(row.document_id)
        query_by_string[row.folded_string] = row.query

    all_strings = sorted(reach_by_string)
    excluded: dict[str, CapState] = {}
    documents_by_string: dict[str, tuple[int, ...]] = {}
    for name in all_strings:
        # A string with no manifest row at all is unknown, not clean. The default
        # is the conservative one on purpose.
        state = states.get(name, CapState.UNKNOWN)
        if state is not CapState.CLEAN:
            excluded[name] = state
            continue
        eligible_docs = tuple(sorted(reach_by_string[name] & with_primary))
        if eligible_docs:
            documents_by_string[name] = eligible_docs

    strings_by_query: dict[str, list[str]] = {}
    for name in sorted(documents_by_string):
        strings_by_query.setdefault(query_by_string[name], []).append(name)

    eligible_documents = {doc for docs in documents_by_string.values() for doc in docs}
    counts = FrameCounts(
        documents_total=len(documents),
        documents_with_primary_section=len(with_primary),
        documents_without_primary_section=len(documents) - len(with_primary),
        strings_total=len(all_strings),
        strings_clean=sum(1 for name in all_strings if states.get(name) is CapState.CLEAN),
        strings_capped=sum(1 for state in excluded.values() if state is CapState.CAPPED),
        strings_unknown=sum(1 for state in excluded.values() if state is CapState.UNKNOWN),
        strings_eligible=len(documents_by_string),
        query_groups_eligible=len(strings_by_query),
        documents_eligible=len(eligible_documents),
    )
    log.info(
        "labeledness.frame.built",
        documents_eligible=counts.documents_eligible,
        strings_eligible=counts.strings_eligible,
        query_groups=counts.query_groups_eligible,
        excluded_capped=counts.strings_capped,
        excluded_unknown=counts.strings_unknown,
    )
    return Frame(
        counts=counts,
        documents_by_string=documents_by_string,
        query_by_string={name: query_by_string[name] for name in documents_by_string},
        strings_by_query={query: tuple(names) for query, names in strings_by_query.items()},
        set_id_by_document={doc: set_ids[doc] for doc in sorted(eligible_documents)},
        excluded_strings=excluded,
    )
