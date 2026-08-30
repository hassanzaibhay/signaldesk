"""openFDA label JSON to typed records.

Pure: no I/O, no ORM, no settings. Everything here is a function of one result
dictionary, so the sectioning rules are testable against committed response
bodies without a database or a network.

The source is loosely typed in ways that matter. Section fields are arrays of
strings, but a label may omit a section entirely, supply an empty array, or
supply an array with blank elements. ``openfda`` is itself optional - unapproved
and some older labels carry no ``openfda`` block at all - and every field inside
it is an array even when it holds one value. None of that is an error; it is the
shape of the corpus, and a parser that raised on it would discard real labels.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from signaldesk.core.errors import IngestError

#: openFDA field name to the section code stored against it. The key order is
#: the order sections are emitted in, so a document's rows are deterministic.
SECTION_FIELDS: dict[str, str] = {
    "boxed_warning": "boxed_warning",
    "warnings": "warnings",
    "warnings_and_cautions": "warnings_and_cautions",
    "adverse_reactions": "adverse_reactions",
}


class SplParseError(IngestError):
    """A result carried no usable identity."""


class LabelSectionRecord(BaseModel):
    """One block of one section."""

    model_config = ConfigDict(frozen=True)

    section_code: str
    ordinal: int
    text: str


class LabelRecord(BaseModel):
    """One label, sectioned."""

    model_config = ConfigDict(frozen=True)

    set_id: str
    spl_id: str = ""
    version: str = ""
    effective_time: str = ""
    brand_names: list[str] = Field(default_factory=list)
    generic_names: list[str] = Field(default_factory=list)
    substance_names: list[str] = Field(default_factory=list)
    rxcuis: list[str] = Field(default_factory=list)
    sections: list[LabelSectionRecord] = Field(default_factory=list)

    @property
    def has_sections(self) -> bool:
        return bool(self.sections)


def _strings(value: Any) -> list[str]:
    """Coerce an openFDA field to a list of non-empty strings.

    Accepts a list, a bare string, or nothing. Blank and whitespace-only
    elements are dropped: they carry no text and would otherwise become empty
    section rows that the retrieval layer has to filter out later.
    """
    if value is None:
        return []
    if isinstance(value, str):
        candidates = [value]
    elif isinstance(value, list):
        candidates = [item for item in value if isinstance(item, str)]
    else:
        return []
    return [stripped for item in candidates if (stripped := item.strip())]


def _first(value: Any) -> str:
    items = _strings(value)
    return items[0] if items else ""


def sections_of(result: dict[str, Any]) -> list[LabelSectionRecord]:
    """Every stored section of one label, in a deterministic order."""
    records: list[LabelSectionRecord] = []
    for field_name, code in SECTION_FIELDS.items():
        for ordinal, text in enumerate(_strings(result.get(field_name))):
            records.append(LabelSectionRecord(section_code=code, ordinal=ordinal, text=text))
    return records


def parse_result(result: dict[str, Any]) -> LabelRecord:
    """One openFDA result to a record.

    ``set_id`` is required and everything else is optional. A result without a
    set id cannot be stored idempotently or revisited, so it is the one field
    whose absence is an error rather than an empty default.
    """
    openfda = result.get("openfda") or {}
    if not isinstance(openfda, dict):
        openfda = {}

    set_id = _first(result.get("set_id"))
    if not set_id:
        message = f"openFDA result has no set_id; keys present: {sorted(result)[:12]}"
        raise SplParseError(message)

    return LabelRecord(
        set_id=set_id,
        spl_id=_first(result.get("id")),
        version=_first(result.get("version")),
        effective_time=_first(result.get("effective_time")),
        brand_names=_strings(openfda.get("brand_name")),
        generic_names=_strings(openfda.get("generic_name")),
        substance_names=_strings(openfda.get("substance_name")),
        rxcuis=_strings(openfda.get("rxcui")),
        sections=sections_of(result),
    )


def parse_results(results: list[dict[str, Any]]) -> list[LabelRecord]:
    """Parse a page of results, keeping the highest version per ``set_id``.

    openFDA can return several revisions of the same label within one search.
    Storing them all would put multiple versions of the same document behind one
    drug string, and the retrieval layer would have to pick between them at read
    time. The revision comparison is on ``effective_time``, which is a
    zero-padded ``YYYYMMDD`` string and therefore orders correctly as text.
    """
    best: dict[str, LabelRecord] = {}
    for result in results:
        record = parse_result(result)
        seen = best.get(record.set_id)
        if seen is None or record.effective_time > seen.effective_time:
            best[record.set_id] = record
    return [best[key] for key in sorted(best)]
