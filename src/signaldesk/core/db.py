"""DuckDB connection factory.

Postgres is the system of record; DuckDB is the analytics engine that reads the
Parquet extracts and builds contingency tables over the full corpus. The two are
bridged by DuckDB's postgres extension rather than by pulling rows through
Python, so a cross-engine join stays in the database.

The database file lives under ``DATA_DIR``, which is a named Docker volume: bind
mounts on Windows are too slow for a file this write-heavy.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import duckdb

from signaldesk.core.config import Settings, get_settings
from signaldesk.core.logging import get_logger

log = get_logger(__name__)

#: Corpus-wide grouping over twenty million cases needs real headroom; four
#: gigabytes is not enough to sort them even with spilling enabled. Still well
#: inside what the container is given, and the engine spills beyond this rather
#: than failing.
DEFAULT_MEMORY_LIMIT = "8GB"
DEFAULT_THREADS = 4


def duckdb_path(settings: Settings | None = None) -> Path:
    """Location of the analytics database file."""
    settings = settings or get_settings()
    return settings.duckdb_path


def connect(
    settings: Settings | None = None,
    *,
    path: Path | None = None,
    read_only: bool = False,
) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection against ``DATA_DIR``.

    Pass ``path`` to point somewhere else (tests use a temporary directory).
    The parent directory is created for writable connections; a read-only
    connection to a database that does not exist is an error worth surfacing.
    """
    settings = settings or get_settings()
    target = path or duckdb_path(settings)
    if not read_only:
        target.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(target), read_only=read_only)
    connection.execute(f"SET memory_limit='{DEFAULT_MEMORY_LIMIT}'")
    connection.execute(f"SET threads={DEFAULT_THREADS}")

    # Somewhere to spill to. Corpus-wide aggregates over tens of millions of
    # rows do not fit in the memory limit, and without a temp directory the
    # engine has nowhere to go but out of memory.
    spill = settings.data_dir / "duckdb-tmp"
    spill.mkdir(parents=True, exist_ok=True)
    connection.execute(f"SET temp_directory='{spill}'")

    # Row order is never meaningful here: every query that matters either
    # aggregates or carries its own ORDER BY. Releasing the guarantee lets the
    # engine spill in operators that otherwise have to stay in memory.
    connection.execute("SET preserve_insertion_order=false")
    return connection


@contextmanager
def connection(
    settings: Settings | None = None,
    *,
    path: Path | None = None,
    read_only: bool = False,
) -> Iterator[duckdb.DuckDBPyConnection]:
    """``connect`` as a context manager, closing the handle on the way out."""
    handle = connect(settings, path=path, read_only=read_only)
    try:
        yield handle
    finally:
        handle.close()


def attach_postgres(
    handle: duckdb.DuckDBPyConnection,
    settings: Settings | None = None,
    *,
    alias: str = "pg",
    read_only: bool = True,
) -> str:
    """Attach the Postgres database so DuckDB can read its tables directly.

    Returns the alias, so callers can write ``f"SELECT * FROM {alias}.faers_case"``.
    Read-only by default: analytics reads from the system of record, it does not
    write to it.
    """
    settings = settings or get_settings()
    handle.execute("INSTALL postgres")
    handle.execute("LOAD postgres")
    suffix = ", READ_ONLY" if read_only else ""
    handle.execute(f"ATTACH '{settings.database_url}' AS {alias} (TYPE POSTGRES{suffix})")
    log.info("duckdb.postgres.attached", alias=alias, read_only=read_only)
    return alias
