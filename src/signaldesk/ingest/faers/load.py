"""Writing a parsed quarter to Parquet and to Postgres.

Parquet is the analytical source of truth. It is partitioned Hive-style by
quarter so the whole corpus is one glob to DuckDB, and re-writing a quarter is a
file replacement rather than a delete-and-insert against a live table.

Postgres holds what the application serves. Rows go in with COPY rather than
INSERT: at tens of millions of rows the difference is hours. Each quarter's rows
are deleted before insert so that a re-run replaces rather than duplicates,
which is what makes the whole pipeline idempotent.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from django.db import connection as django_connection

from signaldesk.core.config import Settings, get_settings
from signaldesk.core.logging import get_logger
from signaldesk.ingest.faers.quarter import Quarter

log = get_logger(__name__)

COPY_BATCH_ROWS = 10_000


@dataclass(frozen=True, slots=True)
class TableTarget:
    """Where a parsed frame goes.

    A target with no `db_table` is analytical only: it lives in Parquet and is
    read through DuckDB. The high-cardinality child tables are all of this kind;
    see docs/adr/0004 for why serving and analytics were split.
    """

    #: Directory name under the Parquet root.
    dataset: str
    #: Postgres table, or None when this dataset is analytical only.
    db_table: str | None
    #: Columns to persist, in order.
    columns: tuple[str, ...]


CASE_TARGET = TableTarget(
    dataset="case",
    db_table="faers_case",
    columns=(
        "primaryid",
        "caseid",
        "caseversion",
        "quarter",
        "fda_dt",
        "fda_dt_raw",
        "fda_dt_precision",
        "event_dt",
        "event_dt_raw",
        "event_dt_precision",
        "sex",
        "age_years",
        "weight_kg",
        "country",
        "country_source",
        "occp_cod",
        "serious",
        "serious_death",
        "outcome_codes",
    ),
)

DRUG_TARGET = TableTarget(
    dataset="drug",
    db_table=None,
    columns=(
        "primaryid",
        "drug_seq",
        "role_cod",
        "drugname_raw",
        "prod_ai",
        "route",
    ),
)

REACTION_TARGET = TableTarget(
    dataset="reaction",
    db_table=None,
    columns=("primaryid", "pt", "drug_rec_act"),
)

OUTCOME_TARGET = TableTarget(dataset="outcome", db_table=None, columns=("primaryid", "outc_cod"))

THERAPY_TARGET = TableTarget(
    dataset="therapy",
    db_table=None,
    columns=(
        "primaryid",
        "dsg_drug_seq",
        "start_dt",
        "start_dt_raw",
        "start_dt_precision",
        "end_dt",
        "end_dt_raw",
        "end_dt_precision",
        "dur_days",
    ),
)

INDICATION_TARGET = TableTarget(
    dataset="indication",
    db_table=None,
    columns=("primaryid", "indi_drug_seq", "indi_pt"),
)

#: Report-source metadata. Ingested and queryable through the analytical store,
#: with no application surface that serves it row by row, so it has no model.
RPSR_TARGET = TableTarget(dataset="report_source", db_table=None, columns=("primaryid", "rpsr_cod"))


def parquet_root(settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    return settings.data_dir / "parquet" / "faers"


def partition_path(target: TableTarget, quarter: Quarter, settings: Settings | None = None) -> Path:
    """Hive-style partition directory for one dataset and quarter."""
    return parquet_root(settings) / target.dataset / f"quarter={quarter.label}"


def write_parquet(
    frame: pl.DataFrame,
    target: TableTarget,
    quarter: Quarter,
    settings: Settings | None = None,
) -> Path:
    """Write one dataset partition, replacing whatever was there."""
    directory = partition_path(target, quarter, settings)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "part-0.parquet"
    present = [column for column in target.columns if column in frame.columns]
    frame.select(present).write_parquet(path, compression="zstd", statistics=True)
    log.debug(
        "faers.parquet.written",
        quarter=quarter.label,
        dataset=target.dataset,
        rows=frame.height,
        bytes=path.stat().st_size,
    )
    return path


def _render(value: object) -> str:
    """Render one value for COPY, distinguishing empty string from null."""
    if value is None:
        return r"\N"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        # JSON columns: the outcome code evidence list.
        return "[" + ",".join(f'"{item}"' for item in value) + "]"
    return str(value)


def copy_into(
    frame: pl.DataFrame,
    target: TableTarget,
    quarter: Quarter,
    settings: Settings | None = None,
    *,
    batch_rows: int = COPY_BATCH_ROWS,
) -> int:
    """Replace a quarter's rows in one Postgres table using COPY.

    Returns the number of rows written. A dataset with no database table is a
    no-op here by design: it lives in the analytical store only.
    """
    if target.db_table is None:
        return 0

    settings = settings or get_settings()
    present = [column for column in target.columns if column in frame.columns]
    columns = ", ".join(present)
    written = 0

    # Django's connection, not a private one. It is the single place database
    # connectivity is configured, and opening a second connection would write to
    # the real database even when the caller is inside a test transaction
    # pointed somewhere else.
    with django_connection.cursor() as wrapper:
        cursor = wrapper.cursor

        # COPY into a staging table first. The same case identifier is
        # republished across quarters, so rows have to be merged on the key
        # rather than appended, and COPY alone cannot express that. The staging
        # table is projected from the target so column types match exactly and
        # its constraints do not come with it: the generated key column has no
        # value yet, and inheriting its NOT NULL would reject every row.
        cursor.execute(
            f"CREATE TEMP TABLE staging AS SELECT {columns} FROM {target.db_table} WITH NO DATA"
        )
        statement = f"COPY staging ({columns}) FROM STDIN WITH (FORMAT csv, NULL '\\N')"
        with cursor.copy(statement) as copy:
            for batch in frame.select(present).iter_slices(n_rows=batch_rows):
                buffer = io.StringIO()
                writer = csv.writer(buffer, lineterminator="\n")
                for row in batch.iter_rows():
                    writer.writerow([_render(value) for value in row])
                copy.write(buffer.getvalue())
                written += batch.height

        if target.dataset == "case":
            # A case republished in a later quarter replaces the earlier copy:
            # the newer publication is the current state of that case.
            assignments = ", ".join(
                f"{column} = EXCLUDED.{column}" for column in present if column != "primaryid"
            )
            cursor.execute(
                f"INSERT INTO {target.db_table} ({columns}) SELECT {columns} FROM staging "
                f"ON CONFLICT (primaryid) DO UPDATE SET {assignments}"
            )
        else:
            # Child rows are replaced wholesale for every case in this batch, so
            # a re-ingest cannot double them up.
            cursor.execute(
                f"DELETE FROM {target.db_table} WHERE primaryid IN "
                "(SELECT DISTINCT primaryid FROM staging)"
            )
            cursor.execute(
                f"INSERT INTO {target.db_table} ({columns}) SELECT {columns} FROM staging"
            )
        cursor.execute("DROP TABLE IF EXISTS staging")

    log.debug("faers.postgres.copied", quarter=quarter.label, table=target.db_table, rows=written)
    return written


def delete_quarter(quarter: Quarter, settings: Settings | None = None) -> None:
    """Remove a quarter's cases from Postgres."""
    with django_connection.cursor() as cursor:
        cursor.execute("DELETE FROM faers_case WHERE quarter = %s", [quarter.label])
