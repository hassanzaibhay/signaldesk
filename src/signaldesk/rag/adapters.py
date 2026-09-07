"""MedCPT, and nothing else.

The whole of this project's dependence on transformers lives in this file, and it
is kept small on purpose. Neither continuous integration job installs the ``ml``
extra, so every line here is a line no CI run will ever execute; the response is
to leave nothing here that could be wrong in an interesting way. Batching, L2
normalisation, the dimension assertion, the pair construction and resumption all
live in ``rag.embed`` and ``rag.index.corpus``, where a stub encoder exercises
them.

What is left is: load a model, tokenize, one forward pass, return an array.

Three models, because MedCPT is asymmetric by design. Its own model card is
explicit that the query encoder is for short texts and the article encoder for
documents, and they are separate checkpoints trained together. Using one for the
other is not a small approximation, it is the wrong model, and it was a defect in
the first version of ``rag.retrieve``.

* ``MedCptArticleEncoder``  - chunks. Takes a text pair; see below.
* ``MedCptQueryEncoder``    - queries. Takes single texts.
* ``MedCptCrossEncoder``    - reranking. Takes (query, text) pairs and returns
  one logit each, because the checkpoint declares a single label.

The article encoder is trained on ``[title, abstract]`` pairs, so it is fed a
pair here too: the section's name and the chunk body. That matches the shape it
was trained on, and it puts "Boxed warning" in front of the text, which is the
distinction the labeledness question turns on. The pair itself is built by
``rag.embed.document_pair``, which is covered.

Verified against the published configs rather than assumed: all three are
``hidden_size`` 768 with ``max_position_embeddings`` 512, and the cross-encoder
declares one label so its logits are ``(n, 1)``. The dimension assertion in
``rag.embed`` still runs on the first batch, because a config is not the weights.

Every class here carries ``pragma: no cover`` on its own line, which excludes the
whole body. That is the honest marker: these bodies are unreachable in CI, and
the count of lines under it is the number worth keeping small.

torch and transformers are reached through ``importlib`` rather than an import
statement. They are in the ``ml`` extra, so they exist in the container and not
in continuous integration, and a plain import makes the type checker say
different things in the two places - a missing module in one, an untyped call in
the other, and an inline ignore that is correct in one and unused in the other. A
module resolved by name is ``Any`` in both, so the gate reports the same thing
wherever it runs, which is the only property that makes it worth having.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any

from signaldesk.stats.types import FloatArray

#: The encoders' positional limit. Longer input is truncated by the tokenizer,
#: which is why the chunker targets well under it and why the benchmark reports
#: how many real chunks would truncate here.
MAX_LENGTH = 512


class _MedCptEncoder:  # pragma: no cover - needs torch, absent from CI
    """Shared loading and forward pass for the two bi-encoder halves.

    They differ only in what they hand the tokenizer - a pair or a single text -
    so the part that differs is one line in each subclass and the part that does
    not is written once.
    """

    def __init__(self, model_id: str, *, max_length: int = MAX_LENGTH) -> None:
        transformers: Any = importlib.import_module("transformers")

        self._model_id = model_id
        self._max_length = max_length
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        self._model = transformers.AutoModel.from_pretrained(model_id).eval()

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def model_revision(self) -> str:
        """The resolved hub commit, or empty when the loader did not report one.

        Best effort. Empty is a valid recorded state meaning the loader did not
        supply it, which is different from the model having no revisions.
        """
        return str(getattr(self._model.config, "_commit_hash", "") or "")

    def token_lengths(self, tokenizer_input: Any) -> list[int]:
        """Real wordpiece length of each input, with no truncation applied.

        Untruncated on purpose: the benchmark needs to know how long the input
        actually is, and a truncated count would report the limit back as though
        nothing had been cut.
        """
        encoded = self._tokenizer(tokenizer_input, truncation=False, padding=False)
        return [len(ids) for ids in encoded["input_ids"]]

    def _forward(self, tokenizer_input: Any) -> FloatArray:
        """Tokenize, one forward pass, take the [CLS] row, return an array.

        [CLS] rather than mean pooling because that is what MedCPT's own card
        does: the representation is the first position's last hidden state.
        """
        torch: Any = importlib.import_module("torch")

        with torch.no_grad():
            encoded = self._tokenizer(
                tokenizer_input,
                truncation=True,
                padding=True,
                max_length=self._max_length,
                return_tensors="pt",
            )
            hidden = self._model(**encoded).last_hidden_state[:, 0, :]
        return hidden.numpy()  # type: ignore[no-any-return]


class MedCptArticleEncoder(_MedCptEncoder):  # pragma: no cover - needs torch
    """Encodes chunks, as (section name, chunk body) pairs."""

    def encode(self, pairs: Sequence[tuple[str, str]]) -> FloatArray:
        return self._forward([list(pair) for pair in pairs])


class MedCptQueryEncoder(_MedCptEncoder):  # pragma: no cover - needs torch
    """Encodes queries, which are single short texts."""

    def encode(self, texts: Sequence[str]) -> FloatArray:
        return self._forward(list(texts))


class MedCptCrossEncoder:  # pragma: no cover - needs torch, absent from CI
    """Scores a query against candidate texts jointly."""

    def __init__(self, model_id: str, *, max_length: int = MAX_LENGTH) -> None:
        transformers: Any = importlib.import_module("transformers")

        self._model_id = model_id
        self._max_length = max_length
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        self._model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_id
        ).eval()

    @property
    def model_id(self) -> str:
        return self._model_id

    def score(self, query: str, texts: Sequence[str]) -> FloatArray:
        """One logit per text. Batching is the caller's job; see rag.embed."""
        torch: Any = importlib.import_module("torch")

        with torch.no_grad():
            encoded = self._tokenizer(
                [[query, text] for text in texts],
                truncation=True,
                padding=True,
                max_length=self._max_length,
                return_tensors="pt",
            )
            # squeeze(1) because the checkpoint declares a single label, so the
            # logits arrive as (n, 1) and the ranking wants (n,).
            logits = self._model(**encoded).logits.squeeze(dim=1)
        return logits.numpy()  # type: ignore[no-any-return]
