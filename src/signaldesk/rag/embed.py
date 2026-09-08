"""Encoder interfaces, and everything around a model call that is not the call.

The two model adapters - the MedCPT article encoder and its cross-encoder - are
deliberately not here. Neither continuous integration job installs the ``ml``
extra, so torch and transformers are absent from every job including the one
that enforces the coverage floor. Anything that imports them is therefore
untestable in CI, and the response is to make that surface as small as it can
be rather than to accept a large one.

So this module holds the protocols and all the logic: batching, L2
normalisation, and the dimension assertion. An adapter's whole job is to turn a
list of strings into an array. Everything that decides anything is here, where a
stub encoder can exercise it.

The dimension assertion earns its place. MedCPT-Article-Encoder is expected to
produce 768 values because it is a PubMedBERT-base model, and that expectation
is not confirmed against the weights. A wrong guess that is checked fails on the
first batch with a message naming both widths. A wrong guess that is not checked
either raises deep inside a database driver or, if something helpfully reshapes,
fills a column with vectors that are not what the column says they are. The
second is the one worth engineering against: an index that is quietly wrong
returns plausible results forever.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from signaldesk.core.errors import SignalDeskError
from signaldesk.rag.device import CPU, surfacing_oom
from signaldesk.stats.types import FloatArray

#: Texts per forward pass when no caller says otherwise. Small because the
#: container has four CPUs and no accelerator, and a large batch there buys
#: nothing while raising peak memory. A caller that resolved a device takes its
#: size from rag.device.encode_batch_size instead, which is the same number on
#: the cpu.
DEFAULT_BATCH_SIZE = 16


class EmbeddingError(SignalDeskError):
    """An encoder produced something the index cannot store."""


@runtime_checkable
class Encoder(Protocol):
    """Turns text into dense vectors.

    Implementations return a ``(len(texts), dimensions)`` array and do nothing
    else - no normalisation, no batching, no validation. Those belong to the
    functions below so that they are exercised by tests that need no model.
    """

    @property
    def model_id(self) -> str:
        """Model identifier as configured, recorded on every row it produces."""
        ...

    @property
    def model_revision(self) -> str:
        """Weights revision, or empty when the loader cannot report one."""
        ...

    def encode(self, texts: Sequence[str]) -> FloatArray:
        """Encode one batch. Row order matches ``texts``."""
        ...


#: What the article encoder is fed: two segments, tokenized as a pair.
TextPair = tuple[str, str]


@runtime_checkable
class DocumentEncoder(Protocol):
    """Turns (section name, chunk body) pairs into dense vectors.

    Separate from ``Encoder`` because MedCPT is asymmetric: the article half is
    trained on two-segment input and the query half on one. Collapsing them into
    a single protocol would let the wrong model be passed to either side without
    anything noticing, which is the defect this split exists to prevent.
    """

    @property
    def model_id(self) -> str: ...

    @property
    def model_revision(self) -> str: ...

    def encode(self, pairs: Sequence[TextPair]) -> FloatArray:
        """Encode one batch of pairs. Row order matches ``pairs``."""
        ...


@runtime_checkable
class CrossEncoder(Protocol):
    """Scores a query against candidate texts jointly.

    Separate from ``Encoder`` because the operation is different in kind: there
    is no reusable vector, every pair costs a forward pass, and the output is a
    relevance score rather than a point in a space.
    """

    @property
    def model_id(self) -> str: ...

    def score(self, query: str, texts: Sequence[str]) -> FloatArray:
        """Relevance of each text to ``query``. Higher is more relevant."""
        ...


def batched[Batchable](items: Sequence[Batchable], size: int) -> Iterator[Sequence[Batchable]]:
    """Split ``items`` into batches of at most ``size``.

    Generic because the two encoder halves batch different things and a
    second copy of this loop for the sake of an element type would be a
    second place for an off-by-one to live.
    """
    if size < 1:
        message = f"batch size must be at least 1, got {size}"
        raise ValueError(message)
    for start in range(0, len(items), size):
        yield items[start : start + size]


def assert_dimensions(vectors: FloatArray, expected: int) -> None:
    """Refuse anything that is not ``(n, expected)``.

    Named in full in the message. "Dimension mismatch" sends a reader to the
    source; "the model produced 384 and the column holds 768" sends them to the
    decision.
    """
    if vectors.ndim != 2:
        message = (
            f"an encoder must return a two-dimensional array of shape "
            f"(texts, {expected}); got shape {vectors.shape}"
        )
        raise EmbeddingError(message)
    produced = vectors.shape[1]
    if produced != expected:
        message = (
            f"the encoder produced {produced}-dimensional vectors and the "
            f"embedding column holds {expected}. This is not a configuration "
            "difference that can be absorbed: storing them would need the "
            "vectors truncated or padded, and either makes the index something "
            "other than what it reports. Change EMBEDDING_DIMENSIONS in "
            "web/documents/models.py and write a migration, or point the "
            "encoder at a model of the declared width."
        )
        raise EmbeddingError(message)


def normalize_rows(vectors: FloatArray) -> FloatArray:
    """Scale every row to unit length.

    The dense index is built with cosine distance over vectors normalised here.
    With unit vectors cosine and inner product produce the same ordering, so the
    stored form is right for either; cosine is the declared operator because it
    stays correct if a normalisation is ever missed, and inner product does not.

    A zero row cannot be normalised and is left as zeros rather than being
    turned into a NaN row that would poison every distance computed against it.
    An encoder returning a zero vector for real text is a defect, but it is one
    the caller can see in the stored row instead of one that silently spreads.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe = np.where(norms == 0.0, 1.0, norms)
    return np.asarray(vectors / safe, dtype=np.float64)


def document_pair(section_label: str, text: str) -> TextPair:
    """The two segments the article encoder is fed for one chunk.

    MedCPT's article encoder is trained on ``[title, abstract]``. A chunk has a
    natural analogue: the section's name and the chunk body. Feeding it that way
    matches the shape the model saw in training, and it puts "Boxed warning" or
    "Adverse reactions" in front of the text, which is exactly the distinction
    the labeledness question turns on.

    The label is the human name, not the storage code: the encoder reads English,
    and ``boxed_warning`` is not a phrase it was trained on.
    """
    return (section_label.strip(), text.strip())


def _validated(vectors_out: object, expected_dimensions: int, batch_length: int) -> FloatArray:
    """One batch's output, checked for width and row count, then normalised."""
    vectors = np.asarray(vectors_out, dtype=np.float64)
    assert_dimensions(vectors, expected_dimensions)
    if vectors.shape[0] != batch_length:
        message = (
            f"the encoder returned {vectors.shape[0]} vectors for {batch_length} "
            "inputs; rows and inputs must correspond by position"
        )
        raise EmbeddingError(message)
    return normalize_rows(vectors)


def embed_texts(
    encoder: Encoder,
    texts: Sequence[str],
    *,
    expected_dimensions: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = CPU,
) -> FloatArray:
    """Encode every text, validated and normalised, in ``texts`` order.

    The assertion runs per batch rather than once at the end, so a mismatch
    costs one batch of work and not the whole corpus.

    ``device`` names the device for the out-of-memory message and nothing else;
    where the encoder actually runs was decided when it was constructed.
    """
    if not texts:
        return np.zeros((0, expected_dimensions), dtype=np.float64)
    with surfacing_oom(batch_size=batch_size, device=device):
        return np.vstack(
            [
                _validated(encoder.encode(batch), expected_dimensions, len(batch))
                for batch in batched(texts, batch_size)
            ]
        )


def embed_pairs(
    encoder: DocumentEncoder,
    pairs: Sequence[TextPair],
    *,
    expected_dimensions: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = CPU,
) -> FloatArray:
    """``embed_texts`` for the two-segment article encoder.

    Same batching, same assertion on the first batch, same normalisation. The
    only difference is what the encoder is handed, which is why the checking is
    shared rather than copied.
    """
    if not pairs:
        return np.zeros((0, expected_dimensions), dtype=np.float64)
    with surfacing_oom(batch_size=batch_size, device=device):
        return np.vstack(
            [
                _validated(encoder.encode(batch), expected_dimensions, len(batch))
                for batch in batched(pairs, batch_size)
            ]
        )


def score_in_batches(
    cross_encoder: CrossEncoder,
    query: str,
    texts: Sequence[str],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = CPU,
) -> FloatArray:
    """Score every text against ``query``, in batches, in ``texts`` order.

    The batching is here rather than in the adapter so that the adapter stays a
    tokenize-and-forward call with nothing in it worth testing. Reranking hands
    over the whole fused pool, which is up to ``dense_top_k + sparse_top_k``
    candidates; one forward pass over all of them at once is a large activation
    for no benefit on a machine with no accelerator.
    """
    if not texts:
        return np.zeros(0, dtype=np.float64)

    blocks = []
    with surfacing_oom(batch_size=batch_size, device=device):
        for batch in batched(texts, batch_size):
            scores = np.asarray(cross_encoder.score(query, batch), dtype=np.float64).reshape(-1)
            if scores.shape[0] != len(batch):
                message = (
                    f"the cross-encoder returned {scores.shape[0]} scores for "
                    f"{len(batch)} texts; scores and texts must correspond by position"
                )
                raise EmbeddingError(message)
            blocks.append(scores)
    return np.concatenate(blocks)
