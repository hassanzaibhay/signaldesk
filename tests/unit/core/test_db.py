"""DuckDB factory: file location, directory creation, and handle lifecycle."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from signaldesk.core import db
from signaldesk.core.config import Settings

pytestmark = pytest.mark.unit


def test_database_lives_under_the_data_directory(settings: Settings) -> None:
    assert db.duckdb_path(settings) == settings.data_dir / "analytics.duckdb"


def test_connect_creates_the_data_directory(settings: Settings) -> None:
    assert not settings.data_dir.exists()
    handle = db.connect(settings)
    try:
        assert settings.data_dir.is_dir()
        assert handle.execute("SELECT 1").fetchone() == (1,)
    finally:
        handle.close()


def test_connect_honours_an_explicit_path(settings: Settings, tmp_path: Path) -> None:
    target = tmp_path / "nested" / "other.duckdb"
    with db.connection(settings, path=target) as handle:
        handle.execute("CREATE TABLE t (x INTEGER)")
        handle.execute("INSERT INTO t VALUES (7)")
        assert handle.execute("SELECT x FROM t").fetchone() == (7,)
    assert target.is_file()


def test_connection_closes_the_handle(settings: Settings) -> None:
    with db.connection(settings) as handle:
        pass
    with pytest.raises(duckdb.ConnectionException):
        handle.execute("SELECT 1")


def test_read_only_connection_to_a_missing_database_fails(settings: Settings) -> None:
    with pytest.raises(duckdb.Error):
        db.connect(settings, path=settings.data_dir / "absent.duckdb", read_only=True)
