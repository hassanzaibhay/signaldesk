"""Reading real extracts, and the file-level irregularities they carry.

The fixture covers both schema eras with genuine public-domain rows. The
encoding and byte-order-mark paths are exercised on constructed files instead,
because the sampled cases happen to be pure ASCII and planting non-ASCII bytes
into a real extract would make the fixture a fake.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from signaldesk.core.errors import IngestError
from signaldesk.ingest.faers.quarter import Quarter
from signaldesk.ingest.faers.reader import (
    detect_encoding,
    locate,
    read_deleted_cases,
    read_table,
)
from signaldesk.ingest.faers.schemas import Table

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "faers"


@pytest.mark.parametrize(
    ("label", "expected_dir"),
    [("2013Q1", "ascii"), ("2013Q2", "ascii"), ("2026Q2", "ASCII")],
)
def test_locates_the_data_directory_whatever_its_case(label: str, expected_dir: str) -> None:
    files = locate(FIXTURES / label, Quarter.parse(label))
    assert files.data_dir.name == expected_dir
    assert set(files.tables) == set(Table)


def test_deleted_file_is_present_only_in_the_later_quarter() -> None:
    assert locate(FIXTURES / "2013Q1", Quarter.parse("2013Q1")).has_deleted_file is False
    assert locate(FIXTURES / "2026Q2", Quarter.parse("2026Q2")).has_deleted_file is True


def test_missing_data_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(IngestError, match="no ascii directory"):
        locate(tmp_path, Quarter.parse("2013Q1"))


@pytest.mark.parametrize(
    ("label", "table", "expected_columns"),
    [
        ("2013Q1", Table.DEMO, 22),
        ("2026Q2", Table.DEMO, 25),
        ("2013Q1", Table.DRUG, 19),
        ("2026Q2", Table.DRUG, 20),
        ("2013Q1", Table.REAC, 3),
        ("2026Q2", Table.REAC, 4),
    ],
)
def test_each_era_reads_with_its_own_column_count(
    label: str, table: Table, expected_columns: int
) -> None:
    quarter = Quarter.parse(label)
    files = locate(FIXTURES / label, quarter)
    frame = read_table(files.tables[table], table, quarter)
    assert frame.width == expected_columns
    assert frame.height > 0


def test_canonical_sex_column_exists_in_both_eras() -> None:
    for label in ("2013Q1", "2026Q2"):
        quarter = Quarter.parse(label)
        files = locate(FIXTURES / label, quarter)
        frame = read_table(files.tables[Table.DEMO], Table.DEMO, quarter)
        assert "sex" in frame.columns
        assert "gndr_cod" not in frame.columns


def test_carriage_returns_do_not_survive_into_the_last_column() -> None:
    quarter = Quarter.parse("2013Q1")
    files = locate(FIXTURES / "2013Q1", quarter)
    frame = read_table(files.tables[Table.DRUG], Table.DRUG, quarter)
    values = [value for value in frame["dose_freq"].to_list() if value is not None]
    assert values
    assert not any(value.endswith("\r") for value in values)


def test_deleted_cases_skips_the_leading_space_line() -> None:
    quarter = Quarter.parse("2026Q2")
    files = locate(FIXTURES / "2026Q2", quarter)
    cases = read_deleted_cases(files.deleted_cases, quarter)
    assert cases
    assert all(isinstance(case, int) for case in cases)


def test_absent_deleted_file_is_an_empty_set_not_an_error() -> None:
    assert read_deleted_cases(None, Quarter.parse("2013Q1")) == set()


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (b"primaryid$caseid\n1$2\n", "ascii"),
        ("primaryid$caseid\n1$naus\u00e9e\n".encode(), "utf-8"),
        ("primaryid$caseid\n1$naus\u00e9e\n".encode("cp1252"), "cp1252"),
    ],
)
def test_encoding_is_detected_per_file(tmp_path: Path, payload: bytes, expected: str) -> None:
    path = tmp_path / "sample.txt"
    path.write_bytes(payload)
    assert detect_encoding(path) == expected


def test_byte_order_mark_does_not_become_part_of_a_column_name(tmp_path: Path) -> None:
    # drug12q4.txt ships with one, which would otherwise rename primaryid.
    quarter = Quarter.parse("2013Q1")
    directory = tmp_path / "ascii"
    directory.mkdir()
    path = directory / "REAC13Q1.txt"
    path.write_bytes("\ufeffprimaryid$caseid$pt\n1$2$NAUSEA\n".encode())
    frame = read_table(path, Table.REAC, quarter)
    assert frame.columns[0] == "primaryid"


def test_a_header_that_does_not_match_its_era_raises(tmp_path: Path) -> None:
    directory = tmp_path / "ascii"
    directory.mkdir()
    path = directory / "OUTC13Q1.txt"
    path.write_bytes(b"primaryid$caseid$outc_code\n1$2$DE\n")
    from signaldesk.core.errors import SchemaMismatchError

    with pytest.raises(SchemaMismatchError, match="outc_cod"):
        read_table(path, Table.OUTC, Quarter.parse("2013Q1"))
