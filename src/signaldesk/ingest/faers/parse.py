"""Normalization of the raw extract into canonical columns.

Three rules govern everything here.

**Nothing is silently coerced.** Every value that cannot be converted becomes
null and is counted under a reason. The counts are reported per quarter, so a
field that degrades upstream shows up as a moving number rather than as a
quietly shrinking denominator.

**Partial dates stay partial.** The source publishes dates as a year, a year and
month, or a full date. A parsed date is stored alongside a precision flag, and
a month-precision value is materialised as the first of the month purely so the
column has a date type. Computing a duration from such a value produces a
fabricated number, so any consumer measuring time-to-onset must filter on
`precision == "day"`. That is a correctness requirement, not a style preference.

**An unknown code raises.** An age unit that is not one of the seven declared is
a schema event: the upstream file changed shape. Treating it as a data-quality
null would silently drop real ages from every age-stratified analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import polars as pl

from signaldesk.core.errors import SchemaMismatchError
from signaldesk.core.logging import get_logger
from signaldesk.ingest.faers.quarter import Quarter
from signaldesk.ingest.faers.schemas import (
    AGE_UNIT_YEARS,
    MAX_PLAUSIBLE_AGE_YEARS,
    MAX_PLAUSIBLE_WEIGHT_KG,
    OUTCOME_DEATH,
    UNKNOWN_COUNTRY_VALUES,
    WEIGHT_UNIT_KG,
)

log = get_logger(__name__)

DATE_PRECISION_DAY = "day"
DATE_PRECISION_MONTH = "month"
DATE_PRECISION_YEAR = "year"
DATE_PRECISION_MISSING = "missing"


@dataclass(slots=True)
class NullCounts:
    """What each normalization discarded, and why.

    Kept as reasons rather than a single total: "8,412 weights were null" is not
    actionable, while "8,410 absent, 1 unparseable, 1 corrupt unit" says whether
    anything needs fixing.
    """

    counts: dict[str, int] = field(default_factory=dict)

    def add(self, reason: str, count: int) -> None:
        if count:
            self.counts[reason] = self.counts.get(reason, 0) + count

    def merge(self, other: NullCounts, prefix: str = "") -> None:
        for reason, count in other.counts.items():
            self.add(f"{prefix}{reason}" if prefix else reason, count)

    def as_dict(self) -> dict[str, int]:
        return dict(sorted(self.counts.items()))


def _as_int(value: Any) -> int:
    """Coerce a polars aggregate to int.

    Aggregations come back as a numeric union that mypy cannot narrow, and every
    use here is a row count. Confined to this one helper so the ``Any`` does not
    spread into a public signature.
    """
    return int(value)


def assert_known_age_units(frame: pl.DataFrame, quarter: Quarter) -> None:
    """Raise when the source introduces an age unit we have not declared."""
    observed = {
        value
        for value in frame.get_column("age_cod").unique().to_list()
        if value is not None and value.strip()
    }
    unknown = sorted(observed - set(AGE_UNIT_YEARS))
    if unknown:
        message = (
            f"{quarter} DEMO: unrecognised age unit code(s) {unknown}. "
            f"declared codes are {sorted(AGE_UNIT_YEARS)}. "
            "this is a schema change and must be declared in schemas.py, not nulled."
        )
        raise SchemaMismatchError(message)


def age_in_years(age: pl.Expr, age_cod: pl.Expr) -> pl.Expr:
    """Convert an age and its unit code into years.

    Implausible results become null; the caller counts them. A missing unit code
    is not assumed to mean years: an unlabelled number could be months, and
    guessing would silently distort the age distribution.
    """
    factor = age_cod.str.strip_chars().replace_strict(
        AGE_UNIT_YEARS, default=None, return_dtype=pl.Float64
    )
    years = age.cast(pl.Float64, strict=False) * factor
    return (
        pl.when(years.is_between(0.0, MAX_PLAUSIBLE_AGE_YEARS, closed="both"))
        .then(years)
        .otherwise(None)
    )


def weight_in_kg(weight: pl.Expr, weight_cod: pl.Expr) -> pl.Expr:
    """Convert a weight and its unit code into kilograms."""
    factor = weight_cod.str.strip_chars().replace_strict(
        WEIGHT_UNIT_KG, default=None, return_dtype=pl.Float64
    )
    kilos = weight.cast(pl.Float64, strict=False) * factor
    return (
        pl.when(kilos.is_between(0.0, MAX_PLAUSIBLE_WEIGHT_KG, closed="none"))
        .then(kilos)
        .otherwise(None)
    )


def date_precision(raw: pl.Expr) -> pl.Expr:
    """Classify a partial date by the length of its digits."""
    cleaned = raw.str.strip_chars()
    length = cleaned.str.len_chars()
    return (
        pl.when(cleaned.is_null() | (length == 0))
        .then(pl.lit(DATE_PRECISION_MISSING))
        .when(length == 8)
        .then(pl.lit(DATE_PRECISION_DAY))
        .when(length == 6)
        .then(pl.lit(DATE_PRECISION_MONTH))
        .when(length == 4)
        .then(pl.lit(DATE_PRECISION_YEAR))
        .otherwise(pl.lit(DATE_PRECISION_MISSING))
    )


def parse_partial_date(raw: pl.Expr) -> pl.Expr:
    """Parse a year, year-month, or full date into a date.

    Year and month precision are filled to the first day of the period so the
    column has a single type. The precision column records which, and nothing
    may compute a duration without consulting it.
    """
    cleaned = raw.str.strip_chars()
    length = cleaned.str.len_chars()
    padded = (
        pl.when(length == 8)
        .then(cleaned)
        .when(length == 6)
        .then(cleaned + pl.lit("01"))
        .when(length == 4)
        .then(cleaned + pl.lit("0101"))
        .otherwise(None)
    )
    return padded.str.to_date(format="%Y%m%d", strict=False)


def normalize_term(term: pl.Expr) -> pl.Expr:
    """Uppercase and collapse internal whitespace, and nothing else.

    These are Preferred Terms arriving from the public extract. Any further
    alteration would break the join to the reference sets, which key on the
    exact term.
    """
    return term.str.strip_chars().str.replace_all(r"\s+", " ").str.to_uppercase().replace("", None)


def normalize_country(value: pl.Expr) -> pl.Expr:
    """Trim and uppercase, and treat a stated unknown as unknown.

    No mapping to ISO codes, which is a later concern. The one substitution is
    turning explicit sentinels such as "COUNTRY NOT SPECIFIED" into null, so that
    deduplication does not block on them; see UNKNOWN_COUNTRY_VALUES.
    """
    cleaned = value.str.strip_chars().str.to_uppercase().replace("", None)
    return pl.when(cleaned.is_in(list(UNKNOWN_COUNTRY_VALUES))).then(None).otherwise(cleaned)


def _date_columns(frame: pl.DataFrame, raw_name: str, base: str) -> pl.DataFrame:
    return frame.with_columns(
        parse_partial_date(pl.col(raw_name)).alias(base),
        date_precision(pl.col(raw_name)).alias(f"{base}_precision"),
    )


def parse_demo(frame: pl.DataFrame, quarter: Quarter) -> tuple[pl.DataFrame, NullCounts]:
    """Normalize the demographics table into case rows."""
    assert_known_age_units(frame, quarter)
    counts = NullCounts()
    supplied_age = _as_int(
        frame.get_column("age").str.strip_chars().replace("", None).is_not_null().sum()
    )
    supplied_weight = _as_int(
        frame.get_column("wt").str.strip_chars().replace("", None).is_not_null().sum()
    )

    corrupt_weight_units = _as_int(
        frame.get_column("wt_cod")
        .str.strip_chars()
        .is_in(list(WEIGHT_UNIT_KG))
        .not_()
        .fill_null(value=False)
        .sum()
    )

    out = frame.with_columns(
        pl.col("primaryid").cast(pl.Int64, strict=False),
        pl.col("caseid").cast(pl.Int64, strict=False),
        pl.col("caseversion").cast(pl.Int64, strict=False),
        age_in_years(pl.col("age"), pl.col("age_cod")).alias("age_years"),
        weight_in_kg(pl.col("wt"), pl.col("wt_cod")).alias("weight_kg"),
        normalize_country(pl.col("occr_country")).alias("occr_country"),
        normalize_country(pl.col("reporter_country")).alias("reporter_country"),
        pl.col("sex").str.strip_chars().str.to_uppercase().replace("", None).alias("sex"),
    )

    # Occurrence country is canonical; reporter country recovers real rows where
    # it is absent. Which one was used is recorded rather than inferred later.
    out = out.with_columns(
        pl.coalesce(pl.col("occr_country"), pl.col("reporter_country")).alias("country"),
        pl.when(pl.col("occr_country").is_not_null())
        .then(pl.lit("occurrence"))
        .when(pl.col("reporter_country").is_not_null())
        .then(pl.lit("reporter"))
        .otherwise(pl.lit("none"))
        .alias("country_source"),
    )

    for raw_name, base in [
        ("event_dt_raw", "event_dt"),
        ("fda_dt_raw", "fda_dt"),
        ("init_fda_dt_raw", "init_fda_dt"),
        ("mfr_dt_raw", "mfr_dt"),
        ("rept_dt_raw", "rept_dt"),
    ]:
        out = _date_columns(out, raw_name, base)

    stated_unknown_country = _as_int(
        frame.get_column("occr_country")
        .str.strip_chars()
        .str.to_uppercase()
        .is_in(list(UNKNOWN_COUNTRY_VALUES))
        .fill_null(value=False)
        .sum()
    )
    unparseable_event_dt = _as_int(
        (
            (out.get_column("event_dt_precision") != DATE_PRECISION_MISSING)
            & out.get_column("event_dt").is_null()
        ).sum()
    )

    converted_age = _as_int(out.get_column("age_years").is_not_null().sum())
    converted_weight = _as_int(out.get_column("weight_kg").is_not_null().sum())

    counts.add("age_missing_or_unconvertible", supplied_age - converted_age)
    counts.add(
        "weight_missing_or_unconvertible",
        supplied_weight - converted_weight - corrupt_weight_units,
    )
    counts.add("weight_implausible_unit", corrupt_weight_units)
    counts.add("sex_missing", _as_int(out.get_column("sex").is_null().sum()))
    counts.add("country_missing", _as_int(out.get_column("country").is_null().sum()))
    counts.add(
        "country_from_reporter_fallback",
        _as_int((out.get_column("country_source") == "reporter").sum()),
    )
    counts.add("country_stated_unknown", stated_unknown_country)
    counts.add("event_dt_unparseable", unparseable_event_dt)
    return out, counts


def derive_seriousness(outcomes: pl.DataFrame) -> pl.DataFrame:
    """Case-level seriousness, derived from the outcome table.

    Definitions, pinned here and in docs/methodology.md so they cannot drift:

    * `serious_death` - the case has an outcome row with code "DE".
    * `serious` - the case has any outcome row at all. Every code this source
      publishes is a seriousness criterion, so presence is the definition and no
      subset is enumerated.

    **A case with no outcome rows is treated as not serious, not as unknown.**
    That is an assumption about the source rather than a fact it states, and it
    is the reason the outcome codes themselves are kept on the case: the derived
    booleans are a convenience, the code list is the evidence.
    """
    return outcomes.group_by("primaryid").agg(
        pl.col("outc_cod").unique().sort().alias("outcome_codes"),
        (pl.col("outc_cod") == OUTCOME_DEATH).any().alias("serious_death"),
        pl.lit(value=True).alias("serious"),
    )


def attach_seriousness(cases: pl.DataFrame, outcomes: pl.DataFrame) -> pl.DataFrame:
    """Join derived seriousness onto cases, defaulting absent outcomes to false."""
    derived = derive_seriousness(outcomes)
    return cases.join(derived, on="primaryid", how="left").with_columns(
        pl.col("serious").fill_null(value=False),
        pl.col("serious_death").fill_null(value=False),
        pl.col("outcome_codes").fill_null(pl.lit([], dtype=pl.List(pl.String))),
    )


def parse_reac(frame: pl.DataFrame, quarter: Quarter) -> tuple[pl.DataFrame, NullCounts]:
    """Normalize reaction terms."""
    counts = NullCounts()
    supplied = _as_int(frame.get_column("pt").is_not_null().sum())
    out = frame.with_columns(
        pl.col("primaryid").cast(pl.Int64, strict=False),
        pl.col("caseid").cast(pl.Int64, strict=False),
        normalize_term(pl.col("pt")).alias("pt"),
    )
    counts.add("pt_missing", supplied - _as_int(out.get_column("pt").is_not_null().sum()))
    log.debug("faers.parse.reac", quarter=quarter.label, rows=out.height)
    return out, counts


def parse_drug(frame: pl.DataFrame, quarter: Quarter) -> tuple[pl.DataFrame, NullCounts]:
    """Normalize the drug table.

    `prod_ai`, the active ingredient the source supplies, only exists from
    2014Q3. Earlier rows get a null column so the Parquet schema is stable across
    eras, and the coverage share is reported rather than assumed.
    """
    counts = NullCounts()
    out = frame.with_columns(
        pl.col("primaryid").cast(pl.Int64, strict=False),
        pl.col("caseid").cast(pl.Int64, strict=False),
        pl.col("drug_seq").cast(pl.Int64, strict=False),
        pl.col("role_cod").str.strip_chars().str.to_uppercase().alias("role_cod"),
        normalize_term(pl.col("drugname_raw")).alias("drugname_norm"),
    )
    if "prod_ai" not in out.columns:
        out = out.with_columns(pl.lit(None, dtype=pl.String).alias("prod_ai"))
    out = out.with_columns(normalize_term(pl.col("prod_ai")).alias("prod_ai"))

    counts.add("drugname_missing", _as_int(out.get_column("drugname_norm").is_null().sum()))
    counts.add("prod_ai_absent", _as_int(out.get_column("prod_ai").is_null().sum()))
    return out, counts


def parse_ther(frame: pl.DataFrame, quarter: Quarter) -> tuple[pl.DataFrame, NullCounts]:
    """Normalize therapy dates, keeping their precision."""
    counts = NullCounts()
    out = frame.with_columns(
        pl.col("primaryid").cast(pl.Int64, strict=False),
        pl.col("caseid").cast(pl.Int64, strict=False),
        pl.col("dsg_drug_seq").cast(pl.Int64, strict=False),
        pl.col("dur").cast(pl.Float64, strict=False),
    )
    for raw_name, base in [("start_dt_raw", "start_dt"), ("end_dt_raw", "end_dt")]:
        out = _date_columns(out, raw_name, base)
    for base in ["start_dt", "end_dt"]:
        counts.add(
            f"{base}_unparseable",
            _as_int(
                (
                    (out.get_column(f"{base}_precision") != DATE_PRECISION_MISSING)
                    & out.get_column(base).is_null()
                ).sum()
            ),
        )
    log.debug("faers.parse.ther", quarter=quarter.label, rows=out.height)
    return out, counts


def parse_simple(
    frame: pl.DataFrame, quarter: Quarter, term_column: str | None = None
) -> tuple[pl.DataFrame, NullCounts]:
    """Normalize the tables that need only identifier casting and a term."""
    counts = NullCounts()
    out = frame.with_columns(
        pl.col("primaryid").cast(pl.Int64, strict=False),
        pl.col("caseid").cast(pl.Int64, strict=False),
    )
    if term_column is not None:
        supplied = _as_int(out.get_column(term_column).is_not_null().sum())
        out = out.with_columns(normalize_term(pl.col(term_column)).alias(term_column))
        counts.add(
            f"{term_column}_missing",
            supplied - _as_int(out.get_column(term_column).is_not_null().sum()),
        )
    log.debug("faers.parse.simple", quarter=quarter.label, rows=out.height)
    return out, counts
