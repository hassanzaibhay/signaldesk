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
from pgvector.django import HnswIndex, VectorField


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


#: Width of the dense vectors this project stores.
#:
#: MedCPT-Article-Encoder is a PubMedBERT-base model, whose hidden size is 768.
#: That is the expectation this column is built against and it has not been
#: confirmed against the weights, which are not a dependency of this app. The
#: embed step asserts the model's real output width against this number and
#: refuses on a mismatch rather than truncating or padding, so a wrong value
#: here fails on the first batch instead of producing an index that is quietly
#: not what it claims. Changing it is a migration, not a setting.
EMBEDDING_DIMENSIONS = 768


class LabelChunk(models.Model):
    """One retrievable window of one label section.

    Deduplicated. A large share of section rows are byte-identical to another:
    the same generic is labelled by many manufacturers and the sections repeat
    verbatim. Storing a chunk per section row would multiply the text and, far
    more expensively, multiply the embeddings computed from it.

    `sha256` covers the section code as well as the text, so identical wording
    under `boxed_warning` and under `adverse_reactions` stays two rows. They are
    two different strengths of claim, and the labeledness question turns on
    which one a sentence came from.
    """

    id = models.BigAutoField(primary_key=True)
    sha256 = models.CharField(max_length=64, unique=True)
    section_code = models.CharField(max_length=24, choices=SectionCode.choices)
    text = models.TextField()
    #: Conservative wordpiece estimate from `rag.chunking`, not tokenizer
    #: output. Stored because the packing decision rested on it and a later
    #: disagreement with the real tokenizer is worth being able to see.
    token_estimate = models.IntegerField()

    class Meta:
        db_table = "label_chunk"
        ordering: ClassVar = ["id"]
        indexes: ClassVar = [
            models.Index(fields=["section_code"], name="label_chunk_section_idx"),
        ]

    def __str__(self) -> str:
        return f"chunk {self.sha256[:12]} ({self.section_code})"


class LabelChunkOccurrence(models.Model):
    """Where one deduplicated chunk appears in the corpus.

    The join that makes deduplication safe. Retrieval returns a chunk; a
    reviewer needs to know which labels, and therefore which drugs, actually say
    it. Without this table a deduplicated chunk has no provenance and the saving
    would have been bought by discarding the thing the corpus is for.
    """

    id = models.BigAutoField(primary_key=True)
    chunk = models.ForeignKey(LabelChunk, on_delete=models.CASCADE, related_name="occurrences")
    section = models.ForeignKey(LabelSection, on_delete=models.CASCADE, related_name="chunks")
    #: Position of this chunk within that section, from zero.
    ordinal = models.IntegerField()

    class Meta:
        db_table = "label_chunk_occurrence"
        ordering: ClassVar = ["section", "ordinal"]
        constraints: ClassVar = [
            models.UniqueConstraint(
                fields=["chunk", "section"], name="label_chunk_occurrence_unique"
            )
        ]
        indexes: ClassVar = [
            models.Index(fields=["section", "ordinal"], name="label_chunk_occ_section_idx"),
        ]

    def __str__(self) -> str:
        return f"{self.chunk_id} in section {self.section_id}[{self.ordinal}]"


class ChunkEmbedding(models.Model):
    """One chunk's dense vector, under one model.

    Separate from the chunk so that chunking and embedding are separable steps:
    chunks exist unembedded, which is the state the corpus is in until the embed
    run happens, and re-embedding under a different encoder adds rows rather
    than rewriting the corpus. The ablation encoder named in the settings needs
    exactly that.

    `model` and `model_revision` are recorded per row rather than assumed from
    configuration. A vector whose producing model cannot be named from the row
    itself is not traceable, and a table holding two models' output with no way
    to tell them apart is worse than one holding neither.
    """

    id = models.BigAutoField(primary_key=True)
    chunk = models.ForeignKey(LabelChunk, on_delete=models.CASCADE, related_name="embeddings")
    #: Model identifier as configured, for example "ncbi/MedCPT-Article-Encoder".
    model = models.CharField(max_length=128)
    #: Weights revision where the loader could report one. Empty means the
    #: loader did not supply it, which is different from the model having no
    #: revisions and is left distinguishable.
    model_revision = models.CharField(max_length=64, blank=True, default="")
    #: The width actually produced, recorded next to the vector rather than
    #: inferred from the column, so a migration that widened the column later
    #: cannot make old rows look like they were always that wide.
    dimensions = models.IntegerField()
    vector = VectorField(dimensions=EMBEDDING_DIMENSIONS)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "chunk_embedding"
        ordering: ClassVar = ["chunk", "model"]
        constraints: ClassVar = [
            models.UniqueConstraint(fields=["chunk", "model"], name="chunk_embedding_unique")
        ]
        indexes: ClassVar = [
            models.Index(fields=["model"], name="chunk_embedding_model_idx"),
            # Cosine, over vectors normalised at write. With unit vectors cosine
            # and inner product rank identically, and cosine stays correct if a
            # normalisation is ever missed, which inner product does not.
            HnswIndex(
                name="chunk_embedding_hnsw",
                fields=["vector"],
                m=16,
                ef_construction=200,
                opclasses=["vector_cosine_ops"],
            ),
        ]

    def __str__(self) -> str:
        return f"{self.chunk_id} under {self.model}"
