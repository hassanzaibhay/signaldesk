"""Building 2x2 contingency tables over the deduplicated case set.

This is the only place that knows how a drug-event pair is counted. The
statistics layer takes ``(a, b, c, d)`` and has no idea where they came from;
everything that decides what those numbers mean is here.

## The population

The unit is the deduplicated case. That is the Parquet case set with two
subtractions applied, both computed by the ingest pass and read from Postgres:

* stage 1, the published rule - only the highest version of each case survives;
* stage 2, probabilistic matching - re-reports of the same event under a
  different case identifier are folded into their canonical case.

Both live in ``faers_duplicate``, and this module removes every ``primaryid``
that appears there. Nothing is recomputed: a second implementation of the
duplicate rule that disagreed with the first by even a few thousand cases would
move every statistic downstream and there would be no way to tell which was
right.

The duplicate list is pulled into memory and registered with DuckDB rather than
reached through the Postgres extension. That extension has to be downloaded on
first use, and an analysis that silently requires the network is an analysis
that cannot be re-run offline. Four and a half million identifiers is 36 MB.

## Drug identity

Two modes, chosen with ``drug_key``:

``ingredient``
    Joins the RxNorm normalization tables. One row per ingredient, so a
    combination product contributes to each of its ingredients. This is the
    analysis ARCHITECTURE section 8.1 specifies and the one whose numbers are
    publishable.

``raw_string``
    Groups on ``upper(trim(drugname_raw))``. Nothing is resolved, so brand and
    generic names for the same substance are different drugs and the counts are
    fragmented across them.

``raw_string`` exists so the engine is testable and measurable before
normalization has been applied to a given database, not as an equivalent
alternative. Which mode produced a number is written into the run parameters and
into the eval artifact, and no number should ever be quoted without it.

## The pair table

One row per ``(drug, PT)`` with at least one co-reporting case. The full cross
product is 641,070 strings by 23,684 terms - fifteen billion cells, almost all
empty - and is never materialised; ``a = 0`` pairs carry no information and the
MGPS likelihood conditions on their absence.

**Every observed pair is returned, including those below the minimum cell
count.** ARCHITECTURE section 8.1 sets that minimum at ``a >= 3`` and says such
pairs are computed and flagged ``insufficient`` rather than dropped, and there
are two independent reasons the filter cannot move earlier than the flag:

* a pair excluded here vanishes from the denominator of every rate reported
  downstream, with nothing left to say it existed;
* MGPS fits its prior on the distribution of counts across the whole table.
  Handing it only the pairs with three or more cases truncates exactly the
  low-count mass the first mixture component exists to model, and the fit
  collapses onto its lower bound rather than converging to something wrong and
  plausible. That failure is reproduced in ``tests/unit/stats/test_mgps.py``.

``min_a`` therefore belongs to the estimator panel, not to this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from signaldesk.analytics.faers import dataset_glob
from signaldesk.core.config import Settings, get_settings
from signaldesk.core.db import connection
from signaldesk.core.errors import SignalDeskError
from signaldesk.core.logging import get_logger
from signaldesk.stats.types import Contingency

if TYPE_CHECKING:
    import duckdb

log = get_logger(__name__)


class DrugKey(StrEnum):
    """What counts as one drug."""

    INGREDIENT = "ingredient"
    RAW_STRING = "raw_string"


class RoleFilter(StrEnum):
    """Which reported drug roles enter the analysis.

    ``PS`` is the primary suspect, ``SS`` a secondary suspect, ``C`` a
    concomitant medication and ``I`` an interacting one. The primary analysis
    takes suspects; the sensitivity analysis takes primary suspects only. Both
    are run and both are reported, because restricting to ``PS`` trades
    sensitivity for specificity and the size of that trade is a finding.
    """

    PS_SS = "ps_ss"
    PS = "ps"

    @property
    def codes(self) -> tuple[str, ...]:
        return ("PS", "SS") if self is RoleFilter.PS_SS else ("PS",)


class ContingencyError(SignalDeskError):
    """The population could not be assembled as asked."""


@dataclass(frozen=True, slots=True)
class ContingencySpec:
    """Everything that decides what the numbers mean. Recorded on the run."""

    drug_key: DrugKey = DrugKey.INGREDIENT
    roles: RoleFilter = RoleFilter.PS_SS
    min_a: int = 3
    start_quarter: str | None = None
    end_quarter: str | None = None

    def as_params(self) -> dict[str, str | int | None]:
        """Flat form for ``signal_run.params`` and the eval artifact."""
        return {
            "drug_key": str(self.drug_key),
            "roles": str(self.roles),
            "role_codes": ",".join(self.roles.codes),
            "min_a": self.min_a,
            "start_quarter": self.start_quarter,
            "end_quarter": self.end_quarter,
        }


@dataclass(frozen=True, slots=True)
class PairTable:
    """The contingency table, with the labels and the population behind it."""

    drug: list[str]
    pt: list[str]
    table: Contingency
    #: Deduplicated cases forming the background.
    n_cases: int
    #: Pairs with at least one co-reporting case. Every one of them is returned.
    pairs_observed: int
    #: Pairs at or above ``spec.min_a``. Reported, not filtered on: the rest are
    #: present in the table and marked ``insufficient`` by the estimator panel.
    pairs_sufficient: int
    #: Cases excluded as duplicates.
    duplicates_removed: int
    #: Cases with no drug row surviving the role filter, or no coded term. They
    #: stay in the background and are reported, not dropped.
    cases_without_drug: int
    cases_without_reaction: int
    spec: ContingencySpec = field(default_factory=ContingencySpec)

    def __len__(self) -> int:
        return len(self.drug)


def _quarter_filter(alias: str, spec: ContingencySpec) -> str:
    clauses = []
    if spec.start_quarter:
        clauses.append(f"{alias}.quarter >= '{spec.start_quarter}'")
    if spec.end_quarter:
        clauses.append(f"{alias}.quarter <= '{spec.end_quarter}'")
    return (" AND " + " AND ".join(clauses)) if clauses else ""


def _duplicate_primaryids() -> pl.DataFrame:
    """Every case identifier the deduplication pass folded into another.

    Read through the ORM's connection so there is one definition of where the
    system of record lives, and returned as a frame so DuckDB can join it
    without a second database driver in the picture.
    """
    from django.db import connection as postgres

    with postgres.cursor() as cursor:
        cursor.execute("SELECT DISTINCT primaryid FROM faers_duplicate")
        rows = [int(row[0]) for row in cursor.fetchall()]
    return pl.DataFrame({"primaryid": rows}, schema={"primaryid": pl.Int64})


def _drug_identity_sql(spec: ContingencySpec, settings: Settings | None) -> str:
    """A query yielding ``(primaryid, drug)`` for the deduplicated population."""
    drug_glob = dataset_glob("drug", settings)
    roles = ", ".join(f"'{code}'" for code in spec.roles.codes)

    if spec.drug_key is DrugKey.RAW_STRING:
        return f"""
            SELECT DISTINCT p.primaryid, upper(trim(d.drugname_raw)) AS drug
            FROM read_parquet('{drug_glob}', hive_partitioning := true) d
            JOIN population p USING (primaryid)
            WHERE d.role_cod IN ({roles}) AND d.drugname_raw IS NOT NULL
              AND trim(d.drugname_raw) <> ''
        """

    # Ingredient mode. The source's own active-ingredient field is preferred
    # over the free-text name where both resolve, because it is already a
    # substance rather than a product; the fallback is recorded in the match
    # table by source_field, not decided here.
    return f"""
        SELECT DISTINCT p.primaryid, i.name AS drug
        FROM read_parquet('{drug_glob}', hive_partitioning := true) d
        JOIN population p USING (primaryid)
        JOIN (
            SELECT m.source_field, m.folded_string, c.name
            FROM drug_string_match m
            JOIN drug_string_ingredient si ON si.match_id = m.id
            JOIN drug_concept c ON c.ingredient_rxcui = si.ingredient_id
        ) i ON i.folded_string = upper(trim(
                   CASE WHEN i.source_field = 'prod_ai' THEN d.prod_ai ELSE d.drugname_raw END
               ))
        WHERE d.role_cod IN ({roles})
    """


#: The mapping tables ingredient mode joins against, and what to read from each.
_NORMALIZATION_QUERIES = {
    "drug_string_match": "SELECT id, source_field, folded_string FROM drug_string_match",
    "drug_string_ingredient": "SELECT match_id, ingredient_id FROM drug_string_ingredient",
    "drug_concept": "SELECT ingredient_rxcui, name FROM drug_concept",
}


def _load_normalization_tables(handle: duckdb.DuckDBPyConnection) -> None:
    """Copy the string-to-ingredient mapping into DuckDB.

    It is small - hundreds of thousands of rows against eighty-four million drug
    rows - so the mapping travels to the analytics engine rather than the corpus
    travelling to Postgres.

    A missing table means normalization has not been applied to this database.
    That is reported as such, naming the command that fixes it, rather than
    surfacing as a driver error about a missing relation: the caller's next
    action is different in the two cases.
    """
    from django.db import ProgrammingError
    from django.db import connection as postgres

    for name, query in _NORMALIZATION_QUERIES.items():
        try:
            with postgres.cursor() as cursor:
                cursor.execute(query)
                columns = [column.name for column in cursor.description or []]
                rows = cursor.fetchall()
        except ProgrammingError as error:
            message = (
                f"ingredient mode needs the {name} table, which does not exist in this "
                "database. Apply the normalization migration and run "
                "'signaldesk normalize drugs', or pass --drug-key raw-string to build "
                "against unresolved drug strings."
            )
            raise ContingencyError(message) from error

        if not rows:
            message = (
                f"ingredient mode needs {name}, which exists but is empty. Run "
                "'signaldesk normalize drugs', or pass --drug-key raw-string."
            )
            raise ContingencyError(message)

        frame = pl.from_records(rows, schema=columns, orient="row")
        handle.register(name, frame)
        log.info("contingency.normalization.loaded", table=name, rows=frame.height)


def build(
    spec: ContingencySpec | None = None,
    settings: Settings | None = None,
) -> PairTable:
    """Assemble the contingency table for every observed drug-event pair."""
    spec = spec or ContingencySpec()
    settings = settings or get_settings()

    case_glob = dataset_glob("case", settings)
    reaction_glob = dataset_glob("reaction", settings)
    duplicates = _duplicate_primaryids()

    with connection(settings) as handle:
        handle.execute("SET enable_progress_bar=false")
        handle.register("faers_duplicate_ids", duplicates)

        # One row per case, not one per publication: the analytical store keeps
        # a row for every quarter a case appeared in. See ADR 0004.
        handle.execute(f"""
            CREATE OR REPLACE TEMP TABLE population AS
            SELECT c.primaryid
            FROM read_parquet('{case_glob}', hive_partitioning := true) c
            ANTI JOIN faers_duplicate_ids f USING (primaryid)
            WHERE 1 = 1 {_quarter_filter("c", spec)}
            QUALIFY row_number() OVER (PARTITION BY c.primaryid ORDER BY c.quarter DESC) = 1
        """)
        n_cases = _scalar(handle, "SELECT count(*) FROM population")
        if n_cases == 0:
            message = "the deduplicated population is empty; nothing to compute"
            raise ContingencyError(message)

        if spec.drug_key is DrugKey.INGREDIENT:
            _load_normalization_tables(handle)

        handle.execute(f"""
            CREATE OR REPLACE TEMP TABLE case_drug AS
            {_drug_identity_sql(spec, settings)}
        """)
        handle.execute(f"""
            CREATE OR REPLACE TEMP TABLE case_reaction AS
            SELECT DISTINCT p.primaryid, upper(trim(r.pt)) AS pt
            FROM read_parquet('{reaction_glob}', hive_partitioning := true) r
            JOIN population p USING (primaryid)
            WHERE r.pt IS NOT NULL AND trim(r.pt) <> ''
        """)

        cases_with_drug = _scalar(handle, "SELECT count(DISTINCT primaryid) FROM case_drug")
        cases_with_reaction = _scalar(handle, "SELECT count(DISTINCT primaryid) FROM case_reaction")

        handle.execute("""
            CREATE OR REPLACE TEMP TABLE drug_margin AS
            SELECT drug, count(DISTINCT primaryid) AS n_drug FROM case_drug GROUP BY drug
        """)
        handle.execute("""
            CREATE OR REPLACE TEMP TABLE pt_margin AS
            SELECT pt, count(DISTINCT primaryid) AS n_pt FROM case_reaction GROUP BY pt
        """)

        # The join is on the case, so a case naming three drugs and two terms
        # contributes six pairs and is counted once in each. That is the
        # definition of the table, not double counting: it is one case for each
        # pair's cell a, and one case in the background for every pair.
        handle.execute("""
            CREATE OR REPLACE TEMP TABLE pair AS
            SELECT d.drug, r.pt, count(*) AS a
            FROM case_drug d JOIN case_reaction r USING (primaryid)
            GROUP BY d.drug, r.pt
        """)
        pairs_observed = _scalar(handle, "SELECT count(*) FROM pair")

        frame = handle.execute(
            """
            SELECT p.drug, p.pt, p.a, dm.n_drug, pm.n_pt
            FROM pair p
            JOIN drug_margin dm USING (drug)
            JOIN pt_margin pm USING (pt)
            ORDER BY p.a DESC, p.drug, p.pt
            """
        ).pl()

    a = frame["a"].to_numpy().astype(np.int64)
    n_drug = frame["n_drug"].to_numpy().astype(np.int64)
    n_pt = frame["n_pt"].to_numpy().astype(np.int64)
    total = np.int64(n_cases)

    table = Contingency(
        a=a,
        b=n_drug - a,
        c=n_pt - a,
        d=total - n_drug - n_pt + a,
    )

    pairs_sufficient = int(np.count_nonzero(a >= spec.min_a))

    log.info(
        "contingency.built",
        drug_key=str(spec.drug_key),
        roles=str(spec.roles),
        cases=n_cases,
        pairs_observed=pairs_observed,
        pairs_sufficient=pairs_sufficient,
    )

    return PairTable(
        drug=frame["drug"].to_list(),
        pt=frame["pt"].to_list(),
        table=table,
        n_cases=n_cases,
        pairs_observed=pairs_observed,
        pairs_sufficient=pairs_sufficient,
        duplicates_removed=duplicates.height,
        cases_without_drug=n_cases - cases_with_drug,
        cases_without_reaction=n_cases - cases_with_reaction,
        spec=spec,
    )


def _scalar(handle: duckdb.DuckDBPyConnection, query: str) -> int:
    row = handle.execute(query).fetchone()
    if row is None:
        message = f"expected one row from {query!r}"
        raise ContingencyError(message)
    return int(row[0])
