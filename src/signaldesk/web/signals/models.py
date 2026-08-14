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
