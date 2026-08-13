"""Typed access to the adverse event data that lives in Parquet.

The five high-cardinality child datasets - drug, reaction, outcome, therapy and
indication - are not Postgres tables. They are Parquet partitions read through
DuckDB, for the reasons recorded in docs/adr/0004. This module is the supported
way to reach them, so that no caller has to hand-write a glob or discover the
partition layout, and so the layout can change without every consumer changing
with it.

Every function here returns aggregates or bounded slices. There is deliberately
no "give me all the drug rows" accessor: nothing in the application serves those
tables row by row, and an unbounded pull of fifty million rows through Python is
the failure mode this split exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from signaldesk.core.config import Settings, get_settings
from signaldesk.core.db import connection
from signaldesk.ingest.faers.load import parquet_root

#: Datasets available in the analytical store.
DATASETS = ("case", "drug", "reaction", "outcome", "therapy", "indication", "report_source")


def dataset_glob(dataset: str, settings: Settings | None = None) -> str:
    """The read_parquet glob for one dataset, partitioned by quarter."""
    if dataset not in DATASETS:
        message = f"unknown dataset {dataset!r}; expected one of {DATASETS}"
        raise ValueError(message)
    return f"{parquet_root(settings)}/{dataset}/*/*.parquet"


def _read(query: str, settings: Settings | None = None) -> pl.DataFrame:
    settings = settings or get_settings()
    with connection(settings) as handle:
        return handle.execute(query).pl()


@dataclass(frozen=True, slots=True)
class CorpusCounts:
    """Row counts per dataset, for the quality report and the corpus page."""

    counts: dict[str, int]

    @property
    def cases(self) -> int:
        return self.counts.get("case", 0)


def corpus_counts(settings: Settings | None = None) -> CorpusCounts:
    """Row counts for every dataset in the analytical store."""
    counts: dict[str, int] = {}
    for dataset in DATASETS:
        frame = _read(
            f"SELECT count(*) AS n FROM read_parquet('{dataset_glob(dataset, settings)}', "
            "hive_partitioning := true)",
            settings,
        )
        counts[dataset] = int(frame.item(0, "n"))
    return CorpusCounts(counts=counts)


def drugs_for_case(primaryid: int, settings: Settings | None = None) -> pl.DataFrame:
    """Every drug row for one case."""
    return _read(
        f"SELECT * FROM read_parquet('{dataset_glob('drug', settings)}', hive_partitioning := true) "
        f"WHERE primaryid = {int(primaryid)} ORDER BY drug_seq",
        settings,
    )


def reactions_for_case(primaryid: int, settings: Settings | None = None) -> pl.DataFrame:
    """Every reaction term for one case."""
    return _read(
        f"SELECT * FROM read_parquet('{dataset_glob('reaction', settings)}', "
        f"hive_partitioning := true) WHERE primaryid = {int(primaryid)} ORDER BY pt",
        settings,
    )


def top_reactions(limit: int = 50, settings: Settings | None = None) -> pl.DataFrame:
    """The most frequently reported terms, by number of cases."""
    return _read(
        f"SELECT pt, count(DISTINCT primaryid) AS cases "
        f"FROM read_parquet('{dataset_glob('reaction', settings)}', hive_partitioning := true) "
        f"WHERE pt IS NOT NULL GROUP BY pt ORDER BY cases DESC LIMIT {int(limit)}",
        settings,
    )


def prod_ai_coverage(settings: Settings | None = None) -> pl.DataFrame:
    """Share of drug rows carrying the source's active ingredient, by quarter.

    The column only exists from 2014Q3, so the early quarters are zero by
    construction. The normalization prompt needs this number to decide how much
    of its coverage can come free from the source.
    """
    return _read(
        "SELECT quarter, count(*) AS rows, "
        "count(prod_ai) AS with_prod_ai, "
        "round(100.0 * count(prod_ai) / count(*), 2) AS pct "
        f"FROM read_parquet('{dataset_glob('drug', settings)}', hive_partitioning := true) "
        "GROUP BY quarter ORDER BY quarter",
        settings,
    )


def case_drug_reaction_pairs(settings: Settings | None = None) -> pl.DataFrame:
    """Distinct (case, drug, reaction) triples, the input to contingency tables.

    Kept here rather than in the statistics layer so that the layout of the
    analytical store stays a detail of this module.
    """
    return _read(
        "SELECT DISTINCT d.primaryid, d.drugname_raw, r.pt "
        f"FROM read_parquet('{dataset_glob('drug', settings)}', hive_partitioning := true) d "
        f"JOIN read_parquet('{dataset_glob('reaction', settings)}', hive_partitioning := true) r "
        "USING (primaryid) "
        "WHERE d.drugname_raw IS NOT NULL AND r.pt IS NOT NULL",
        settings,
    )
