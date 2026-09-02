"""The annotation store: append-only, flushed per verdict, undone by tombstone.

A day of annotation is the most expensive artifact this project produces and the
only one that cannot be regenerated. Everything here is shaped by that.

**Append-only.** The file is never seeked backwards into and no record is ever
rewritten. Undo appends a tombstone naming the screen it retracts, so a crash in
the middle of an undo leaves a file that is still a valid prefix of itself.

**Flushed and fsynced per verdict**, not per screen and not at exit. The cost is
one sync per keypress, which is nothing next to the seconds a screen takes.

**Resolved by replay.** The live verdict for a screen is the last record naming
it. Readers replay the file in order; nothing derives state from anywhere else,
so what a resumed session believes is exactly what is on disk.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

from signaldesk.core.errors import AnnotationError
from signaldesk.core.logging import get_logger
from signaldesk.evals.labeledness.manifest import Protocol

log = get_logger(__name__)


class Verdict(StrEnum):
    """The five verdicts. See guideline section 3."""

    LABELLED_EXPLICIT = "l"
    LABELLED_BROADER = "b"
    LABELLED_CLASS = "c"
    NOT_LABELLED = "n"
    UNCLEAR = "u"

    @property
    def is_labelled(self) -> bool:
        """The binary collapse the metric is computed on."""
        return self in _LABELLED

    @property
    def label(self) -> str:
        return _LABELS[self]


_LABELLED: frozenset[Verdict] = frozenset(
    {Verdict.LABELLED_EXPLICIT, Verdict.LABELLED_BROADER, Verdict.LABELLED_CLASS}
)

_LABELS: dict[Verdict, str] = {
    Verdict.LABELLED_EXPLICIT: "labelled, explicit",
    Verdict.LABELLED_BROADER: "labelled, broader term",
    Verdict.LABELLED_CLASS: "labelled, class warning",
    Verdict.NOT_LABELLED: "not labelled",
    Verdict.UNCLEAR: "unclear",
}


class Record(BaseModel):
    """One line of the store: a verdict, or a tombstone retracting one."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["verdict", "undo"]
    screen_id: str
    recorded_at: datetime
    guideline_version: str

    verdict: Verdict | None = None
    note: str | None = None
    protocol: Protocol | None = None
    elapsed_ms: int | None = None

    #: What was adjudicated against, carried on the record and not only in the
    #: manifest, so a record stays self-describing if the two are separated.
    set_id: str | None = None
    document_id: int | None = None
    section_codes: tuple[str, ...] = ()
    digests: dict[str, str] = {}

    @model_validator(mode="after")
    def _verdicts_are_complete(self) -> Self:
        if self.kind == "undo":
            if self.verdict is not None:
                message = "an undo record must not carry a verdict"
                raise ValueError(message)
            return self
        if self.verdict is None:
            message = "a verdict record must carry a verdict"
            raise ValueError(message)
        if self.set_id is None or self.document_id is None or not self.digests:
            message = (
                "a verdict record must carry the document and the digests it was "
                "adjudicated against"
            )
            raise ValueError(message)
        return self


class AnnotationStore:
    """Append-only JSONL over one gold-set file."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, record: Record) -> None:
        """Write one record and return only once it is on the platter.

        ``flush`` moves it out of the Python buffer and ``fsync`` out of the
        kernel's. Without the second, a machine that loses power keeps the
        verdicts the page cache had not written, which is the failure this whole
        module exists to prevent.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def read_all(self) -> list[Record]:
        """Every record in file order, verdicts and tombstones alike."""
        if not self.path.exists():
            return []
        records: list[Record] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for number, raw in enumerate(handle, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    records.append(Record.model_validate_json(line))
                except ValueError as exc:
                    message = f"{self.path}:{number} is not a valid annotation record: {exc}"
                    raise AnnotationError(message) from exc
        return records

    def resolved(self) -> dict[str, Record]:
        """Screen id -> its live verdict record. Tombstoned screens are absent."""
        return resolve(self.read_all())


def resolve(records: Sequence[Record]) -> dict[str, Record]:
    """Replay the log. Last record naming a screen wins; an undo removes it."""
    live: dict[str, Record] = {}
    for record in records:
        if record.kind == "undo":
            live.pop(record.screen_id, None)
        else:
            live[record.screen_id] = record
    return live


def next_position(manifest_length: int, live: dict[str, Record], screen_ids: Sequence[str]) -> int:
    """The 1-indexed position to resume at: the first screen with no live verdict.

    Scans from the front rather than continuing from the highest answered
    position, so a screen retracted by an undo three sessions ago is picked up
    again instead of being skipped forever.
    """
    for index, screen_id in enumerate(screen_ids, start=1):
        if screen_id not in live:
            return index
    return manifest_length + 1


def verdict_record(
    *,
    screen_id: str,
    verdict: Verdict,
    guideline_version: str,
    protocol: Protocol,
    elapsed_ms: int,
    set_id: str,
    document_id: int,
    section_codes: Sequence[str],
    digests: dict[str, str],
    note: str | None = None,
) -> Record:
    return Record(
        kind="verdict",
        screen_id=screen_id,
        recorded_at=datetime.now(tz=UTC),
        guideline_version=guideline_version,
        verdict=verdict,
        note=note,
        protocol=protocol,
        elapsed_ms=elapsed_ms,
        set_id=set_id,
        document_id=document_id,
        section_codes=tuple(section_codes),
        digests=dict(digests),
    )


def undo_record(*, screen_id: str, guideline_version: str) -> Record:
    return Record(
        kind="undo",
        screen_id=screen_id,
        recorded_at=datetime.now(tz=UTC),
        guideline_version=guideline_version,
    )


def iter_verdicts(records: Sequence[Record]) -> Iterator[Record]:
    """The live verdicts only, tombstoned screens already removed."""
    yield from resolve(records).values()
