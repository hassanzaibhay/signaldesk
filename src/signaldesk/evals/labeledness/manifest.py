"""The committed sample manifest: every screen, and every byte of it.

The manifest carries the full section text rather than a pointer to it, and that
is the design decision the rest of the annotation path rests on.

* The harness needs no database. Annotation reads one JSON file, so a corpus
  refresh mid-annotation cannot change the stimulus underneath a verdict, and
  the text that was judged is the text that is committed by construction rather
  than by anyone remembering to keep them in step.
* A verdict stays re-checkable. Years later the exact rendered text is still
  there, without re-deriving it from a corpus that has moved.

It is written with ``ensure_ascii=True``. Label text is very close to pure ASCII
already -- escaping costs about half a percent on this corpus -- so the whole
artifact sits inside hygiene rule 1 without needing an allowlist exemption, and
under rule 3 it cannot sit untracked.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from signaldesk.core.errors import AnnotationError
from signaldesk.evals.labeledness.frame import FrameCounts

ARTIFACT_KIND: Final = "labeledness_sample"

#: Adverse-reactions text above this many characters gets the bounded reading
#: protocol from guideline section 6.2. Just above the 75th percentile (11,116)
#: of the 8,878 eligible documents, whose median is 6,450.
LONG_SECTION_CHARS: Final = 12_000


class Protocol(StrEnum):
    """Which reading protocol the guideline puts a screen on."""

    FULL = "full"
    BOUNDED = "bounded"


def protocol_for(primary_chars: int) -> Protocol:
    """Decide the protocol from the length of the primary section, before display.

    Decided at draw time and committed, not decided at render time, so the
    protocol on a record cannot drift from the protocol the annotator was shown.
    """
    return Protocol.BOUNDED if primary_chars > LONG_SECTION_CHARS else Protocol.FULL


def digest(text: str) -> str:
    """SHA-256 of one block of section text, as rendered."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class SectionBlock(BaseModel):
    """One block of one section, with the digest of exactly these bytes."""

    model_config = ConfigDict(frozen=True)

    section_code: str
    ordinal: int
    text: str
    sha256: str

    @classmethod
    def build(cls, *, section_code: str, ordinal: int, text: str) -> SectionBlock:
        return cls(section_code=section_code, ordinal=ordinal, text=text, sha256=digest(text))

    @model_validator(mode="after")
    def _digest_matches(self) -> Self:
        if self.sha256 != digest(self.text):
            message = (
                f"section {self.section_code}[{self.ordinal}] carries a digest that does "
                "not match its text; the manifest has been edited after it was written"
            )
            raise ValueError(message)
        return self


class Screen(BaseModel):
    """One annotation screen: a drug string, an event, and one document.

    ``repeat_of`` is in the manifest because scoring needs it and is never in
    anything the harness renders. :func:`render.render_screen` takes a
    :class:`Screen` and does not read this field; the test that pins it renders a
    repeat twice, once with ``repeat_of`` cleared, and compares the bytes.
    """

    model_config = ConfigDict(frozen=True)

    screen_id: str
    position: int
    #: Stable across the two presentations of a repeat. Agreement groups on it.
    pair_id: str
    #: The FAERS string as the signal table keys it, shown verbatim.
    drug_string: str
    #: The openFDA query the string was fetched under. The sibling group key.
    query: str
    pt: str
    set_id: str
    document_id: int
    protocol: Protocol
    primary_chars: int
    sections: tuple[SectionBlock, ...]
    is_repeat: bool = False
    repeat_of: str | None = None

    @property
    def section_codes(self) -> tuple[str, ...]:
        seen: list[str] = []
        for block in self.sections:
            if block.section_code not in seen:
                seen.append(block.section_code)
        return tuple(seen)

    def digests(self) -> dict[str, str]:
        """Block key -> digest, the shape an annotation record stores."""
        return {f"{block.section_code}[{block.ordinal}]": block.sha256 for block in self.sections}


class SampleManifest(BaseModel):
    """The committed frame, seed, provenance and screen list."""

    model_config = ConfigDict(frozen=True)

    artifact: Literal["labeledness_sample"] = ARTIFACT_KIND
    run_id: str
    written_at: datetime
    seed: int
    #: The version the sample was DRAWN under. A record carries the version its
    #: verdict was MADE under, and after an amendment the two differ. This field
    #: is never rewritten to match a later guideline; that would claim the sample
    #: was drawn against a document that did not exist when it was drawn.
    guideline_version: str
    long_section_chars: int = LONG_SECTION_CHARS

    #: Provenance. The SPL ingest artifact does not record which signal run its
    #: scope was selected from -- a gap the README lists -- so recording both
    #: here, together with the single-partition check, is what joins them.
    signal_run_id: str
    signal_artifact: str
    spl_artifact: str
    signal_partitions_on_disk: int

    frame: FrameCounts
    frame_pairs: int

    screens: tuple[Screen, ...]
    reserve: tuple[Screen, ...] = ()
    reserve_note: str = ""
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _positions_are_dense_and_ordered(self) -> Self:
        expected = list(range(1, len(self.screens) + 1))
        if [screen.position for screen in self.screens] != expected:
            message = "screens must be ordered by position with no gaps"
            raise ValueError(message)
        ids = [screen.screen_id for screen in self.screens]
        if len(set(ids)) != len(ids):
            message = "screen ids must be unique"
            raise ValueError(message)
        return self

    def screen_at(self, position: int) -> Screen:
        return self.screens[position - 1]


def write_manifest(manifest: SampleManifest, path: Path) -> int:
    """Write the manifest as pure-ASCII JSON. Returns the byte count."""
    payload = manifest.model_dump(mode="json")
    text = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if not text.isascii():
        message = "manifest is not ASCII after ensure_ascii; refusing to write"
        raise AnnotationError(message)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="ascii", newline="\n")
    return len(text)


def read_manifest(path: Path) -> SampleManifest:
    """Read and validate a committed manifest.

    Validation is not a formality here: :class:`SectionBlock` re-derives every
    digest, so a manifest edited after it was committed fails to load rather than
    scoring against text nobody saw.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        message = f"cannot read sample manifest {path}: {exc}"
        raise AnnotationError(message) from exc
    try:
        return SampleManifest.model_validate(payload)
    except ValueError as exc:
        message = f"sample manifest {path} is malformed: {exc}"
        raise AnnotationError(message) from exc


def manifest_filename(run_id: str) -> str:
    return f"labeledness_sample_{run_id}.json"


def new_run_id() -> str:
    return datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")


class ReserveNote(BaseModel):
    """Why the reserve stratum exists, carried in the artifact itself."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(
        default=(
            "The reserve is a pre-registered stratum of pairs carrying explicit "
            "lexical evidence, committed with the primary draw so that it cannot "
            "be chosen after seeing results. It is insurance for work after this "
            "deadline, not a rescue for this run. If the primary sample returns "
            "few labelled pairs the honest response is to report the labelled "
            "rate with its confidence interval and state the limitation, not to "
            "annotate the reserve under time pressure on the last day. An "
            "unannotated reserve is the expected state and is not an omission."
        )
    )
