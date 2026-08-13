"""Locating and reading the files inside an extracted quarter.

Every irregularity observed across the corpus is absorbed here, so that no
caller needs a conditional about it:

* The data directory is `ascii/` in some quarters and `ASCII/` in others, which
  only matters on a case-sensitive filesystem, which is where this runs.
* Line endings are CRLF in some files and LF in others, sometimes within the
  same archive.
* Encoding is per file, not per archive: `DRUG13Q1.txt` is UTF-8 while its
  siblings in the same zip are pure ASCII. A single archive-wide assumption
  fails on one file in hundreds, which is the worst kind of failure because it
  looks like a data problem rather than a reader problem.
* Fields are `$`-separated and are not quoted. Drug names contain bare quote
  characters, so quote processing has to be switched off or rows silently merge.
"""

from __future__ import annotations

import codecs
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import polars as pl

from signaldesk.core.errors import IngestError
from signaldesk.core.logging import get_logger
from signaldesk.ingest.faers.quarter import Quarter
from signaldesk.ingest.faers.schemas import (
    DELIMITER,
    Table,
    TableSchema,
    schema_for,
    validate_header,
)

log = get_logger(__name__)

#: Tried in order. ASCII is a subset of UTF-8, so a file that decodes as ASCII
#: also decodes as UTF-8; it is listed separately only so the log records which
#: files actually carry non-ASCII bytes. cp1252 is the practical last resort for
#: Windows-authored text and, unlike latin-1, is not a decode-anything fallback
#: that would hide a genuine encoding surprise.
ENCODING_ORDER: tuple[str, ...] = ("ascii", "utf-8", "cp1252")

CHUNK_BYTES = 1 << 20


@dataclass(frozen=True, slots=True)
class QuarterFiles:
    """Where the parts of an extracted quarter actually are."""

    quarter: Quarter
    data_dir: Path
    tables: dict[Table, Path]
    deleted_cases: Path | None

    @property
    def has_deleted_file(self) -> bool:
        """Whether this quarter shipped a deleted-cases list at all.

        Absent is not the same as empty, and the difference is reported: early
        quarters ship no such file, so their deletions are unknown rather than
        known to be zero.
        """
        return self.deleted_cases is not None


def _find_dir(root: Path, name: str) -> Path | None:
    for candidate in root.rglob("*"):
        if candidate.is_dir() and candidate.name.lower() == name:
            return candidate
    return None


def locate(root: Path, quarter: Quarter) -> QuarterFiles:
    """Find the data directory, the seven table files, and any deleted list."""
    data_dir = _find_dir(root, "ascii")
    if data_dir is None:
        message = f"{quarter}: no ascii directory in the extracted archive at {root}"
        raise IngestError(message)

    tables: dict[Table, Path] = {}
    for table in Table:
        matches = [
            path
            for path in data_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() == ".txt"
            and path.name.upper().startswith(table.value)
        ]
        if not matches:
            message = f"{quarter}: no {table.value} file in {data_dir}"
            raise IngestError(message)
        if len(matches) > 1:
            names = sorted(path.name for path in matches)
            message = f"{quarter}: {len(matches)} candidate {table.value} files: {names}"
            raise IngestError(message)
        tables[table] = matches[0]

    deleted_dir = _find_dir(root, "deleted")
    deleted_file: Path | None = None
    if deleted_dir is not None:
        candidates = [path for path in deleted_dir.iterdir() if path.suffix.lower() == ".txt"]
        deleted_file = candidates[0] if len(candidates) == 1 else None

    return QuarterFiles(
        quarter=quarter, data_dir=data_dir, tables=tables, deleted_cases=deleted_file
    )


def detect_encoding(path: Path) -> str:
    """The first declared encoding that decodes the whole file.

    Decoded incrementally so a 150 MB file does not have to be held in memory
    twice to answer the question.
    """
    for encoding in ENCODING_ORDER:
        decoder = codecs.getincrementaldecoder(encoding)(errors="strict")
        try:
            with path.open("rb") as handle:
                while chunk := handle.read(CHUNK_BYTES):
                    decoder.decode(chunk)
                decoder.decode(b"", final=True)
        except UnicodeDecodeError:
            continue
        return encoding
    message = f"{path.name}: does not decode as any of {ENCODING_ORDER}"
    raise IngestError(message)


def _as_utf8(path: Path, encoding: str) -> Path | BytesIO:
    """Hand polars a UTF-8 source, transcoding only when it has to."""
    if encoding in {"ascii", "utf-8"}:
        return path
    return BytesIO(path.read_bytes().decode(encoding).encode("utf-8"))


def read_table(path: Path, table: Table, quarter: Quarter) -> pl.DataFrame:
    """Read one table file, validate its header, and rename to canonical names.

    Every column is read as text. These extracts put "YEARS" in a weight-unit
    column and free text in numeric ones, so type conversion belongs in the
    normalization step where a failure can be counted rather than in the reader
    where it would either raise or coerce silently.
    """
    schema: TableSchema = schema_for(table, quarter)
    encoding = detect_encoding(path)
    source = _as_utf8(path, encoding)

    frame = pl.read_csv(
        source,
        separator=DELIMITER,
        has_header=True,
        quote_char=None,
        infer_schema_length=0,
        truncate_ragged_lines=True,
    )

    # Some quarters pad header fields with spaces, a CRLF file leaves a carriage
    # return on the last one, and drug12q4.txt carries a byte order mark that
    # would otherwise turn its first column into a BOM-prefixed name. Normalise
    # names once, here, so validation and renaming agree on what a column is
    # called.
    frame = frame.rename({name: name.strip().lstrip("\ufeff") for name in frame.columns})
    validate_header(schema, list(frame.columns), quarter)

    # A CRLF file leaves a trailing carriage return on the final field; the
    # header check passes because it is stripped there, so strip the data too.
    last = schema.canonical_names[-1]
    frame = frame.rename(schema.rename_map).with_columns(
        pl.col(last).str.strip_chars_end("\r").alias(last)
    )

    log.debug(
        "faers.read.table",
        quarter=quarter.label,
        table=table.value,
        rows=frame.height,
        encoding=encoding,
        file=path.name,
    )
    return frame


def read_deleted_cases(path: Path | None, quarter: Quarter) -> set[int]:
    """Case identifiers withdrawn by the source in this quarter.

    The file is a bare list with no header, and its first line is a single
    space, so anything that is not a number is skipped rather than parsed. A
    quarter that ships no such file yields an empty set; the caller records that
    the file was absent, which is different from it being empty.
    """
    if path is None:
        return set()

    encoding = detect_encoding(path)
    cases: set[int] = set()
    skipped = 0
    for line in path.read_text(encoding=encoding).splitlines():
        token = line.strip()
        if not token:
            continue
        if not token.isdigit():
            skipped += 1
            continue
        cases.add(int(token))

    log.info(
        "faers.read.deleted_cases",
        quarter=quarter.label,
        cases=len(cases),
        skipped_lines=skipped,
        file=path.name,
    )
    return cases
