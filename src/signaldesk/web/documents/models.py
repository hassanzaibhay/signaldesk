"""Structured product labels, sectioned, and the keys that reach them.

Three tables, and the shape is dictated by the question the retrieval layer asks
of them: given a drug string that carries a signal, what does its label already
say about this reaction. That is a lookup from a FAERS drug string to a set of
label sections, so the string is a first-class key here rather than something
resolved at query time.

* `LabelDocument` is one SPL, identified by `set_id`. openFDA republishes a
  label under a new `spl_id` for every revision while `set_id` stays put, so
  `set_id` is the identity and `version` is what moves. Storing per `spl_id`
  would accumulate a new row per revision and make "the current label" a query
  rather than a fact.
* `LabelSection` is the sectioned text, one row per section occurrence. openFDA
  returns each section as an array of strings, and the elements are not
  interchangeable prose - they are separate blocks - so `ordinal` preserves the
  published order rather than joining them into one field.
* `LabelDrugKey` is the reach-through. One row per (drug string, document),
  carrying which route found it. Without this table, answering the retrieval
  question means re-running the openFDA query at read time.

`ingredient_rxcui` is a plain integer and deliberately not a foreign key to
`signals.DrugConcept`. A cross-app foreign key would tie this app's migration
graph to the signals app's, and the roadmap runs those as parallel tracks with
disjoint file ownership precisely so that does not happen. The cost is that a
dangling rxcui is possible; the check belongs in the ingest, not in the schema.

Raw section text only. Chunking, embedding and indexing belong to the retrieval
prompt and are not pre-empted here.
"""

from __future__ import annotations

from typing import ClassVar

from django.db import models


class SectionCode(models.TextChoices):
    """The label sections this project reads.

    Four of the many an SPL can carry. These are the ones that state what the
    manufacturer has already acknowledged about harm, which is the only thing
    the labeledness question needs. `warnings` and `warnings_and_cautions` are
    both present because the two coexist across label eras: the older format
    uses `warnings`, the Physician Labeling Rule format uses
    `warnings_and_cautions`, and a corpus spanning both has some of each.
    """

    BOXED_WARNING = "boxed_warning", "Boxed warning"
    WARNINGS = "warnings", "Warnings"
    WARNINGS_AND_CAUTIONS = "warnings_and_cautions", "Warnings and cautions"
    ADVERSE_REACTIONS = "adverse_reactions", "Adverse reactions"


class MatchRoute(models.TextChoices):
    """How a drug string was turned into an openFDA query."""

    INGREDIENT = "ingredient", "RxNorm ingredient rxcui"
    CLEANED_STRING = "cleaned_string", "Deterministically cleaned drug string"
    OVERRIDE = "override", "Curated override"


class LabelDocument(models.Model):
    """One structured product label, at the version openFDA currently serves."""

    id = models.BigAutoField(primary_key=True)
    #: Stable across revisions. The identity of "this label".
    set_id = models.CharField(max_length=64, unique=True)
    #: The specific revision openFDA returned. Moves when the label is revised.
    spl_id = models.CharField(max_length=64, blank=True, default="")
    version = models.CharField(max_length=16, blank=True, default="")
    #: As published, `YYYYMMDD`. Kept as the source string rather than parsed:
    #: it is an identifier of a revision here, not a date anything is computed
    #: from, and parsing it would invent precision the field does not carry.
    effective_time = models.CharField(max_length=16, blank=True, default="")

    brand_names = models.JSONField(default=list)
    generic_names = models.JSONField(default=list)
    substance_names = models.JSONField(default=list)
    #: Every rxcui openFDA associates with this label, not only the one queried.
    rxcuis = models.JSONField(default=list)

    retrieved_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "label_document"
        ordering: ClassVar = ["set_id"]
        indexes: ClassVar = [
            models.Index(fields=["effective_time"], name="label_doc_effective_idx"),
        ]

    def __str__(self) -> str:
        primary = self.brand_names[0] if self.brand_names else self.set_id
        return f"{primary} ({self.set_id})"


class LabelSection(models.Model):
    """One block of one section of one label."""

    id = models.BigAutoField(primary_key=True)
    document = models.ForeignKey(LabelDocument, on_delete=models.CASCADE, related_name="sections")
    section_code = models.CharField(max_length=24, choices=SectionCode.choices)
    #: Position within that section's published array. Order is meaning.
    ordinal = models.IntegerField(default=0)
    text = models.TextField()

    class Meta:
        db_table = "label_section"
        ordering: ClassVar = ["document", "section_code", "ordinal"]
        constraints: ClassVar = [
            models.UniqueConstraint(
                fields=["document", "section_code", "ordinal"], name="label_section_unique"
            )
        ]
        indexes: ClassVar = [
            models.Index(fields=["section_code"], name="label_section_code_idx"),
        ]

    def __str__(self) -> str:
        return f"{self.document_id}/{self.section_code}[{self.ordinal}]"


class LabelDrugKey(models.Model):
    """A FAERS drug string, and a label reached from it.

    Many-to-many in practice: one string can match several labels (a substance
    with many marketed products), and one label can be reached from several
    strings. Both directions are indexed because both are queried - the brief
    asks "labels for this string", and the eval asks "strings behind this
    label".
    """

    id = models.BigAutoField(primary_key=True)
    #: `upper(trim(drugname_raw))`, the form the signal table is keyed on.
    folded_string = models.TextField()
    #: What was actually sent to openFDA, so the chain is recoverable.
    query = models.TextField()
    route = models.CharField(max_length=16, choices=MatchRoute.choices)
    #: Null unless the ingredient route found it. Not a foreign key; see module
    #: docstring.
    ingredient_rxcui = models.BigIntegerField(null=True)
    document = models.ForeignKey(LabelDocument, on_delete=models.CASCADE, related_name="drug_keys")

    class Meta:
        db_table = "label_drug_key"
        ordering: ClassVar = ["folded_string", "document"]
        constraints: ClassVar = [
            models.UniqueConstraint(
                fields=["folded_string", "document"], name="label_drug_key_unique"
            )
        ]
        indexes: ClassVar = [
            models.Index(fields=["folded_string"], name="label_drug_key_string_idx"),
            models.Index(fields=["ingredient_rxcui"], name="label_drug_key_rxcui_idx"),
        ]

    def __str__(self) -> str:
        return f"{self.folded_string} -> {self.document_id} ({self.route})"
