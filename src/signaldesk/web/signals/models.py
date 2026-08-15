"""Adverse event case tables.

The shape follows ARCHITECTURE section 7.1. Three deliberate departures from a
naive transcription of the source files, each recorded because a reviewer will
ask:

* `serious` and `serious_death` do not exist in the source. They are derived
  from the outcome table, and the outcome codes are kept on the case as well so
  the derivation can be audited and so a serious-only sensitivity analysis has
  the evidence rather than the conclusion.
* Dates are stored three times over: the raw string exactly as published, a
  parsed date, and a precision flag. The source publishes partial dates, and a
  duration computed from a year-only value padded to January is a fabricated
  number.
* `country` is the occurrence country where present and the reporter country
  otherwise, with `country_source` recording which. Occurrence country has
  substantial nulls and the fallback recovers real records.

No table partitioning in v1. Roughly 14 million case rows with the indexes
declared here is well within a single table, and partitioning would add
migration complexity for no measured benefit.

Only the case, its duplicate judgements, and the ingest manifest live here. The
five high-cardinality child tables - drug, reaction, outcome, therapy and
indication - are held in Parquet and read through DuckDB; see docs/adr/0004.
Reach for `signaldesk.analytics.faers` rather than a model that is not here.
"""

from __future__ import annotations

from typing import ClassVar

from django.db import models


class DatePrecision(models.TextChoices):
    """How much of a partial date the source actually supplied."""

    DAY = "day", "Day"
    MONTH = "month", "Month"
    YEAR = "year", "Year"
    MISSING = "missing", "Missing"


class CountrySource(models.TextChoices):
    """Which column the canonical country came from."""

    OCCURRENCE = "occurrence", "Occurrence country"
    REPORTER = "reporter", "Reporter country"
    NONE = "none", "Neither supplied"


class Case(models.Model):
    """One adverse event case, at its highest version within the corpus."""

    primaryid = models.BigIntegerField(primary_key=True)
    caseid = models.BigIntegerField()
    caseversion = models.IntegerField(null=True)
    quarter = models.CharField(max_length=6)

    fda_dt = models.DateField(null=True)
    #: Exactly as published. Null means the source supplied nothing, which is
    #: different from an empty string and has to stay distinguishable.
    fda_dt_raw = models.CharField(max_length=16, null=True)
    fda_dt_precision = models.CharField(
        max_length=8, choices=DatePrecision.choices, default=DatePrecision.MISSING
    )
    event_dt = models.DateField(null=True)
    event_dt_raw = models.CharField(max_length=16, null=True)
    event_dt_precision = models.CharField(
        max_length=8, choices=DatePrecision.choices, default=DatePrecision.MISSING
    )

    sex = models.CharField(max_length=8, null=True)
    age_years = models.FloatField(null=True)
    weight_kg = models.FloatField(null=True)
    country = models.CharField(max_length=32, null=True)
    country_source = models.CharField(
        max_length=12, choices=CountrySource.choices, default=CountrySource.NONE
    )
    occp_cod = models.CharField(max_length=16, null=True)

    serious = models.BooleanField(default=False)
    serious_death = models.BooleanField(default=False)
    #: The outcome codes the two booleans were derived from. Evidence, not
    #: decoration: an absent list means the case had no outcome rows.
    outcome_codes = models.JSONField(default=list)

    class Meta:
        db_table = "faers_case"
        ordering: ClassVar = ["-fda_dt", "primaryid"]
        indexes: ClassVar = [
            models.Index(fields=["quarter"], name="faers_case_quarter_idx"),
            models.Index(fields=["fda_dt"], name="faers_case_fda_dt_idx"),
            models.Index(fields=["caseid"], name="faers_case_caseid_idx"),
        ]

    def __str__(self) -> str:
        return f"case {self.caseid} v{self.caseversion} ({self.primaryid})"


class Duplicate(models.Model):
    """A case judged to duplicate another, with the evidence for the judgement.

    Written by the corpus-wide deduplication pass, which truncates and rebuilds
    this table rather than appending, so it always reflects one coherent run.
    """

    id = models.BigAutoField(primary_key=True)
    primaryid = models.BigIntegerField()
    canonical_primaryid = models.BigIntegerField()
    method = models.CharField(max_length=32)
    score = models.FloatField(null=True)
    drug_jaccard = models.FloatField(null=True)
    reaction_jaccard = models.FloatField(null=True)
    cross_quarter = models.BooleanField(default=False)

    class Meta:
        db_table = "faers_duplicate"
        constraints: ClassVar = [
            models.UniqueConstraint(fields=["primaryid", "method"], name="faers_duplicate_unique")
        ]
        indexes: ClassVar = [
            models.Index(fields=["canonical_primaryid"], name="faers_dup_canonical_idx"),
        ]

    def __str__(self) -> str:
        return f"{self.primaryid} -> {self.canonical_primaryid} ({self.method})"


class DrugConcept(models.Model):
    """An RxNorm ingredient concept, as the statistics will group by it.

    Ingredient level, `IN` or `PIN`. Everything downstream is computed per
    ingredient, so this is the vocabulary the whole analysis is expressed in.
    """

    ingredient_rxcui = models.BigIntegerField(primary_key=True)
    name = models.CharField(max_length=255)
    tty = models.CharField(max_length=8)
    #: Populated only if the ATC lookup is scoped into this prompt; null means
    #: not fetched, which is different from no ATC class existing.
    atc_code = models.CharField(max_length=16, null=True)

    class Meta:
        db_table = "drug_concept"
        ordering: ClassVar = ["name"]

    def __str__(self) -> str:
        return f"{self.name} ({self.ingredient_rxcui})"


class DrugStringMatch(models.Model):
    """What one distinct source string resolved to.

    Keyed on the string rather than on the drug row. There are 84,589,952 drug
    rows and 656,589 distinct folded strings across the two source fields, and
    the mapping is a property of the string: storing it per row would multiply
    the same judgement 129 times over and make re-running normalization a rewrite
    of the corpus rather than of the mapping. Per-row resolution is a join, in
    priority order, in DuckDB. See docs/adr/0004 for why the drug rows are not
    here to be joined in Postgres.

    `folded_string` is `upper(trim(x))`, which is the form the distinct-string
    counts were measured on and the form the join back to Parquet applies.
    `cleaned_string` is what was actually sent to RxNav, so the chain from
    published string to query is recoverable.
    """

    class SourceField(models.TextChoices):
        PROD_AI = "prod_ai", "Active ingredient field"
        DRUGNAME = "drugname", "Drug name field"

    class Method(models.TextChoices):
        OVERRIDE = "override", "Curated override"
        OVERRIDE_NOT_A_DRUG = "override_not_a_drug", "Curated: not a drug"
        EXACT = "exact", "Exact term match"
        APPROXIMATE = "approximate", "Approximate term match"
        UNMATCHED = "unmatched", "No match above the floor"

    id = models.BigAutoField(primary_key=True)
    source_field = models.CharField(max_length=16, choices=SourceField.choices)
    folded_string = models.TextField()
    cleaned_string = models.TextField()
    #: The concept the string matched, which is not necessarily an ingredient:
    #: a brand or a clinical drug resolves to one through `related`.
    rxcui = models.BigIntegerField(null=True)
    match_method = models.CharField(max_length=24, choices=Method.choices)
    #: RxNav's approximate score, recorded because it was returned, not because
    #: it is a confidence. It tracks query length; see docs/adr/0008.
    match_score = models.FloatField(null=True)
    #: Which candidate resolved to an ingredient. Rank 1 can resolve to none.
    candidate_rank = models.IntegerField(null=True)
    #: Components a combination string was split into, and how many resolved.
    #: A partially resolved combination is a shortened ingredient set, so the
    #: shortfall is recorded rather than left to be inferred from a count of
    #: associated rows.
    components_total = models.IntegerField(default=1)
    components_matched = models.IntegerField(default=0)
    retrieved_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "drug_string_match"
        ordering: ClassVar = ["source_field", "folded_string"]
        constraints: ClassVar = [
            models.UniqueConstraint(
                fields=["source_field", "folded_string"], name="drug_string_match_unique"
            )
        ]
        indexes: ClassVar = [
            models.Index(fields=["match_method"], name="drug_string_method_idx"),
        ]

    def __str__(self) -> str:
        return f"{self.folded_string} -> {self.rxcui} ({self.match_method})"


class DrugStringIngredient(models.Model):
    """One ingredient a string resolved to, of possibly several.

    A combination product is several ingredients, and picking one of them would
    be a fabricated simplification: the whole set is recorded, in the order the
    components appeared in the string.
    """

    id = models.BigAutoField(primary_key=True)
    match = models.ForeignKey(DrugStringMatch, on_delete=models.CASCADE, related_name="ingredients")
    ingredient = models.ForeignKey(
        DrugConcept, on_delete=models.PROTECT, related_name="string_matches"
    )
    ordinal = models.IntegerField(default=0)

    class Meta:
        db_table = "drug_string_ingredient"
        ordering: ClassVar = ["match", "ordinal"]
        constraints: ClassVar = [
            models.UniqueConstraint(
                fields=["match", "ingredient"], name="drug_string_ingredient_unique"
            )
        ]

    def __str__(self) -> str:
        return f"{self.match_id} -> {self.ingredient_id}"


class IngestManifest(models.Model):
    """One row per ingested unit, making the pipeline resumable and idempotent.

    Keyed on source and unit. A completed unit whose upstream checksum still
    matches is skipped; one whose checksum changed is re-ingested and the change
    is logged rather than ignored.
    """

    class Status(models.TextChoices):
        RUNNING = "running", "Running"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"

    id = models.BigAutoField(primary_key=True)
    source = models.CharField(max_length=32)
    unit = models.CharField(max_length=32)
    checksum = models.CharField(max_length=64, blank=True, default="")
    row_count = models.BigIntegerField(default=0)
    row_counts = models.JSONField(default=dict)
    bytes_downloaded = models.BigIntegerField(default=0)
    had_deleted_file = models.BooleanField(default=False)
    status = models.CharField(max_length=12, choices=Status.choices, default=Status.RUNNING)
    error = models.TextField(blank=True, default="")
    started_at = models.DateTimeField(auto_now_add=True)
    finished_at = models.DateTimeField(null=True)

    class Meta:
        db_table = "ingest_manifest"
        ordering: ClassVar = ["source", "unit"]
        constraints: ClassVar = [
            models.UniqueConstraint(fields=["source", "unit"], name="ingest_manifest_unique")
        ]

    def __str__(self) -> str:
        return f"{self.source}/{self.unit}: {self.status}"
