"""Declared column sets, one per schema era, per table.

Nothing here is inferred at runtime. A quarter's header is checked against the
declaration for its era and a mismatch raises, because a column set that shifted
upstream is a schema event: silently accepting it would mean loading a column
into the wrong field and reporting statistics computed on the wrong data.

The two eras were derived by reading headers either side of the boundary rather
than from documentation:

    2014Q1  DEMO 22 cols (gndr_cod)   DRUG 19 cols            REAC 3 cols
    2014Q2  DEMO 22 cols (gndr_cod)   DRUG 19 cols            REAC 3 cols
    2014Q3  DEMO 25 cols (sex)        DRUG 20 (+prod_ai)      REAC 4 (+drug_rec_act)
    2014Q4  DEMO 25 cols (sex)        DRUG 20 (+prod_ai)      REAC 4 (+drug_rec_act)

Canonical names are era-independent. The sex column is `gndr_cod` before the
boundary and `sex` after it, and that difference lives here as a mapping rather
than as a conditional in the parser.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import polars as pl

from signaldesk.core.errors import SchemaMismatchError
from signaldesk.ingest.faers.quarter import Quarter

#: A polars dtype, which may be given as the class or as an instance.
PolarsDataType = type[pl.DataType] | pl.DataType

#: The first quarter of the current era, established from the headers above.
ERA_BOUNDARY = Quarter(2014, 3)

#: The first published quarter, which spells two columns unlike its successors.
FIRST_QUARTER = Quarter(2012, 4)

#: Field separator used by every file in every quarter observed.
DELIMITER = "$"


class Era(StrEnum):
    """Which column set a quarter uses.

    Three, not two. The first published quarter spells two columns differently
    from every quarter after it, which was found by the header check rejecting
    it rather than by any documentation.
    """

    FIRST = "2012q4_only"
    ORIGINAL = "pre_2014q3"
    CURRENT = "2014q3_onward"


class Table(StrEnum):
    """The seven files shipped in each archive, by their filename prefix."""

    DEMO = "DEMO"
    DRUG = "DRUG"
    INDI = "INDI"
    OUTC = "OUTC"
    REAC = "REAC"
    RPSR = "RPSR"
    THER = "THER"


def era_for(quarter: Quarter) -> Era:
    """The schema era a quarter belongs to."""
    if quarter >= ERA_BOUNDARY:
        return Era.CURRENT
    if quarter == FIRST_QUARTER:
        return Era.FIRST
    return Era.ORIGINAL


@dataclass(frozen=True, slots=True)
class Column:
    """One source column, its era-independent name, and its target type.

    Files are read as text and converted afterwards: these are dirty extracts
    where an age column contains "YEARS" and a weight unit contains an age unit,
    so letting a CSV reader guess types would either fail or coerce silently.
    `dtype` is what the value becomes once normalized, not how it is read.
    """

    source: str
    canonical: str
    dtype: PolarsDataType


@dataclass(frozen=True, slots=True)
class TableSchema:
    """The declared column set for one table in one era."""

    table: Table
    era: Era
    columns: tuple[Column, ...]

    @property
    def source_names(self) -> tuple[str, ...]:
        return tuple(column.source for column in self.columns)

    @property
    def canonical_names(self) -> tuple[str, ...]:
        return tuple(column.canonical for column in self.columns)

    @property
    def rename_map(self) -> dict[str, str]:
        return {column.source: column.canonical for column in self.columns}


def _cols(*specs: tuple[str, str, PolarsDataType]) -> tuple[Column, ...]:
    return tuple(
        Column(source=source, canonical=canonical, dtype=dtype)
        for source, canonical, dtype in specs
    )


_TEXT = pl.String
_INT = pl.Int64
_FLOAT = pl.Float64

# --- demographics -----------------------------------------------------------

_DEMO_COMMON_HEAD = (
    ("primaryid", "primaryid", _INT),
    ("caseid", "caseid", _INT),
    ("caseversion", "caseversion", _INT),
    ("i_f_code", "i_f_code", _TEXT),
    ("event_dt", "event_dt_raw", _TEXT),
    ("mfr_dt", "mfr_dt_raw", _TEXT),
    ("init_fda_dt", "init_fda_dt_raw", _TEXT),
    ("fda_dt", "fda_dt_raw", _TEXT),
    ("rept_cod", "rept_cod", _TEXT),
)

_DEMO_COMMON_TAIL = (
    ("e_sub", "e_sub", _TEXT),
    ("wt", "wt", _FLOAT),
    ("wt_cod", "wt_cod", _TEXT),
    ("rept_dt", "rept_dt_raw", _TEXT),
    ("to_mfr", "to_mfr", _TEXT),
    ("occp_cod", "occp_cod", _TEXT),
    ("reporter_country", "reporter_country", _TEXT),
    ("occr_country", "occr_country", _TEXT),
)

DEMO_ORIGINAL = TableSchema(
    table=Table.DEMO,
    era=Era.ORIGINAL,
    columns=_cols(
        *_DEMO_COMMON_HEAD,
        ("mfr_num", "mfr_num", _TEXT),
        ("mfr_sndr", "mfr_sndr", _TEXT),
        ("age", "age", _FLOAT),
        ("age_cod", "age_cod", _TEXT),
        # The era difference that matters: this column is named `sex` after the
        # boundary and carries the same values.
        ("gndr_cod", "sex", _TEXT),
        *_DEMO_COMMON_TAIL,
    ),
)

DEMO_CURRENT = TableSchema(
    table=Table.DEMO,
    era=Era.CURRENT,
    columns=_cols(
        *_DEMO_COMMON_HEAD,
        ("auth_num", "auth_num", _TEXT),
        ("mfr_num", "mfr_num", _TEXT),
        ("mfr_sndr", "mfr_sndr", _TEXT),
        ("lit_ref", "lit_ref", _TEXT),
        ("age", "age", _FLOAT),
        ("age_cod", "age_cod", _TEXT),
        ("age_grp", "age_grp", _TEXT),
        ("sex", "sex", _TEXT),
        *_DEMO_COMMON_TAIL,
    ),
)

# --- drugs ------------------------------------------------------------------

_DRUG_TAIL = (
    ("val_vbm", "val_vbm", _TEXT),
    ("route", "route", _TEXT),
    ("dose_vbm", "dose_vbm", _TEXT),
    ("cum_dose_chr", "cum_dose_chr", _TEXT),
    ("cum_dose_unit", "cum_dose_unit", _TEXT),
    ("dechal", "dechal", _TEXT),
    ("rechal", "rechal", _TEXT),
    ("lot_num", "lot_num", _TEXT),
    ("exp_dt", "exp_dt_raw", _TEXT),
    ("nda_num", "nda_num", _TEXT),
    ("dose_amt", "dose_amt", _TEXT),
    ("dose_unit", "dose_unit", _TEXT),
    ("dose_form", "dose_form", _TEXT),
    ("dose_freq", "dose_freq", _TEXT),
)

DRUG_ORIGINAL = TableSchema(
    table=Table.DRUG,
    era=Era.ORIGINAL,
    columns=_cols(
        ("primaryid", "primaryid", _INT),
        ("caseid", "caseid", _INT),
        ("drug_seq", "drug_seq", _INT),
        ("role_cod", "role_cod", _TEXT),
        ("drugname", "drugname_raw", _TEXT),
        *_DRUG_TAIL,
    ),
)

DRUG_CURRENT = TableSchema(
    table=Table.DRUG,
    era=Era.CURRENT,
    columns=_cols(
        ("primaryid", "primaryid", _INT),
        ("caseid", "caseid", _INT),
        ("drug_seq", "drug_seq", _INT),
        ("role_cod", "role_cod", _TEXT),
        ("drugname", "drugname_raw", _TEXT),
        # Added at the boundary. Null for every row before it, which is a
        # coverage fact the normalization prompt depends on.
        ("prod_ai", "prod_ai", _TEXT),
        *_DRUG_TAIL,
    ),
)

# --- the tables that did not change ----------------------------------------

_INDI_COLUMNS = _cols(
    ("primaryid", "primaryid", _INT),
    ("caseid", "caseid", _INT),
    ("indi_drug_seq", "indi_drug_seq", _INT),
    ("indi_pt", "indi_pt", _TEXT),
)

_OUTC_COLUMNS = _cols(
    ("primaryid", "primaryid", _INT),
    ("caseid", "caseid", _INT),
    ("outc_cod", "outc_cod", _TEXT),
)

_RPSR_COLUMNS = _cols(
    ("primaryid", "primaryid", _INT),
    ("caseid", "caseid", _INT),
    ("rpsr_cod", "rpsr_cod", _TEXT),
)

_THER_COLUMNS = _cols(
    ("primaryid", "primaryid", _INT),
    ("caseid", "caseid", _INT),
    ("dsg_drug_seq", "dsg_drug_seq", _INT),
    ("start_dt", "start_dt_raw", _TEXT),
    ("end_dt", "end_dt_raw", _TEXT),
    ("dur", "dur", _FLOAT),
    ("dur_cod", "dur_cod", _TEXT),
)

REAC_ORIGINAL = TableSchema(
    table=Table.REAC,
    era=Era.ORIGINAL,
    columns=_cols(
        ("primaryid", "primaryid", _INT),
        ("caseid", "caseid", _INT),
        ("pt", "pt", _TEXT),
    ),
)

REAC_CURRENT = TableSchema(
    table=Table.REAC,
    era=Era.CURRENT,
    columns=_cols(
        ("primaryid", "primaryid", _INT),
        ("caseid", "caseid", _INT),
        ("pt", "pt", _TEXT),
        ("drug_rec_act", "drug_rec_act", _TEXT),
    ),
)

#: 2012Q4 spells the lot number column `lot_nbr`. The canonical name is the same
#: as everywhere else, so nothing downstream knows this quarter is different.
DRUG_FIRST = TableSchema(
    table=Table.DRUG,
    era=Era.FIRST,
    columns=_cols(
        ("primaryid", "primaryid", _INT),
        ("caseid", "caseid", _INT),
        ("drug_seq", "drug_seq", _INT),
        ("role_cod", "role_cod", _TEXT),
        ("drugname", "drugname_raw", _TEXT),
        ("val_vbm", "val_vbm", _TEXT),
        ("route", "route", _TEXT),
        ("dose_vbm", "dose_vbm", _TEXT),
        ("cum_dose_chr", "cum_dose_chr", _TEXT),
        ("cum_dose_unit", "cum_dose_unit", _TEXT),
        ("dechal", "dechal", _TEXT),
        ("rechal", "rechal", _TEXT),
        ("lot_nbr", "lot_num", _TEXT),
        ("exp_dt", "exp_dt_raw", _TEXT),
        ("nda_num", "nda_num", _TEXT),
        ("dose_amt", "dose_amt", _TEXT),
        ("dose_unit", "dose_unit", _TEXT),
        ("dose_form", "dose_form", _TEXT),
        ("dose_freq", "dose_freq", _TEXT),
    ),
)

#: 2012Q4 spells the outcome column `outc_code`.
OUTC_FIRST = TableSchema(
    table=Table.OUTC,
    era=Era.FIRST,
    columns=_cols(
        ("primaryid", "primaryid", _INT),
        ("caseid", "caseid", _INT),
        ("outc_code", "outc_cod", _TEXT),
    ),
)

SCHEMAS: dict[tuple[Table, Era], TableSchema] = {
    (Table.DEMO, Era.FIRST): TableSchema(
        table=Table.DEMO, era=Era.FIRST, columns=DEMO_ORIGINAL.columns
    ),
    (Table.DRUG, Era.FIRST): DRUG_FIRST,
    (Table.REAC, Era.FIRST): TableSchema(
        table=Table.REAC, era=Era.FIRST, columns=REAC_ORIGINAL.columns
    ),
    (Table.OUTC, Era.FIRST): OUTC_FIRST,
    (Table.DEMO, Era.ORIGINAL): DEMO_ORIGINAL,
    (Table.DEMO, Era.CURRENT): DEMO_CURRENT,
    (Table.DRUG, Era.ORIGINAL): DRUG_ORIGINAL,
    (Table.DRUG, Era.CURRENT): DRUG_CURRENT,
    (Table.REAC, Era.ORIGINAL): REAC_ORIGINAL,
    (Table.REAC, Era.CURRENT): REAC_CURRENT,
    **{
        (table, era): TableSchema(table=table, era=era, columns=columns)
        for table, columns in [
            (Table.INDI, _INDI_COLUMNS),
            (Table.OUTC, _OUTC_COLUMNS),
            (Table.RPSR, _RPSR_COLUMNS),
            (Table.THER, _THER_COLUMNS),
        ]
        for era in Era
        if (table, era) != (Table.OUTC, Era.FIRST)
    },
}


def schema_for(table: Table, quarter: Quarter) -> TableSchema:
    """The declared schema for a table in a quarter."""
    return SCHEMAS[(table, era_for(quarter))]


#: Age unit codes and their value in years. A code outside this set raises rather
#: than nulling: it means the upstream file changed, and a silently nulled age
#: would disappear from every age-stratified analysis without trace.
#:
#: Seven of these were found by sampling 2013Q1, 2013Q2 and 2026Q2. `SEC` was
#: found by this rule firing on 2012Q4, which is the argument for the rule.
AGE_UNIT_YEARS: dict[str, float] = {
    "DEC": 10.0,
    "YR": 1.0,
    "MON": 1.0 / 12.0,
    "WK": 1.0 / 52.1775,
    "DY": 1.0 / 365.25,
    "HR": 1.0 / 8766.0,
    "MIN": 1.0 / 525960.0,
    "SEC": 1.0 / 31_557_600.0,
}

#: Weight unit codes and their value in kilograms. `YEARS` is observed in this
#: column in one row of 2013Q1; it is corrupt rather than convertible and is
#: counted separately from ordinary missing weights.
WEIGHT_UNIT_KG: dict[str, float] = {
    "KG": 1.0,
    "LBS": 0.45359237,
    "GMS": 0.001,
}

#: Biologically implausible above this, and the source contains no legitimate
#: value near it: the oldest observed age in the sampled quarters is 113 years.
MAX_PLAUSIBLE_AGE_YEARS = 120.0

#: Above the heaviest recorded human by a wide margin.
MAX_PLAUSIBLE_WEIGHT_KG = 650.0

#: Outcome code meaning the patient died. Used to derive `serious_death`.
OUTCOME_DEATH = "DE"

#: Values that state, in words, that the country is unknown. They are normalized
#: to null rather than kept, because deduplication blocks on country: left as a
#: value, every record carrying this sentinel would land in one block whose only
#: shared property is absent information, and comparisons inside it would be
#: matching noise. This is not a mapping of country names, which stays out of
#: ingest; it is recognising an explicit statement of absence.
UNKNOWN_COUNTRY_VALUES: frozenset[str] = frozenset({"COUNTRY NOT SPECIFIED"})


def validate_header(schema: TableSchema, header: list[str], quarter: Quarter) -> None:
    """Check a file's header against its declared column set.

    Raises with both directions of the difference and the positions, because
    "columns do not match" without saying which way is a bug report that takes an
    hour to act on.
    """
    found = tuple(field.strip() for field in header)
    expected = schema.source_names
    if found == expected:
        return

    missing = [name for name in expected if name not in found]
    unexpected = [name for name in found if name not in expected]
    detail = (
        f"{quarter} {schema.table.value}: header does not match the {schema.era.value} schema. "
        f"expected {len(expected)} columns, found {len(found)}."
    )
    if missing:
        detail += f" missing: {missing}."
    if unexpected:
        detail += f" unexpected: {unexpected}."
    if not missing and not unexpected:
        detail += f" same names in a different order: expected {expected}, found {found}."
    raise SchemaMismatchError(detail)
