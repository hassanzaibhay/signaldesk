"""Shared test fixtures.

Unit tests construct settings explicitly with ``_env_file=None`` so that a local
``.env`` cannot change what they assert.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from signaldesk.core.config import Settings

SettingsFactory = Callable[..., Settings]


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every settings key from the environment.

    A test that asserts a default has to run without one, and the container
    exports most of these from its .env file.
    """
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)


@pytest.fixture
def make_settings(tmp_path: Path) -> SettingsFactory:
    """Build a Settings object isolated from the environment and the filesystem."""

    def factory(**overrides: object) -> Settings:
        values: dict[str, object] = {
            "django_secret_key": "test-only-not-a-secret",
            "data_dir": tmp_path / "data",
            "cache_dir": tmp_path / "cache",
            "model_dir": tmp_path / "models",
        }
        values.update(overrides)
        return Settings(_env_file=None, **values)

    return factory


@pytest.fixture
def settings(make_settings: SettingsFactory) -> Settings:
    """A ready-made isolated Settings object."""
    return make_settings()


# ---------------------------------------------------------------------------
# Retrieval test doubles.
#
# Neither continuous integration job installs the ``ml`` extra, so no test may
# import torch or transformers. These stand in for the two model adapters.
#
# They are not random. A hashing vectoriser gives a query that shares words with
# a chunk a genuinely higher cosine similarity than one that does not, and the
# overlap scorer ranks the same way, so a test can assert that retrieval put the
# right chunk first rather than only that it returned the right number of rows.
# A random-vector stub would prove the plumbing and nothing about the ranking.
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class HashingEncoder:
    """Feature-hashing stand-in for a dense encoder."""

    def __init__(self, dimensions: int = 768, model_id: str = "stub/hashing-encoder") -> None:
        self._dimensions = dimensions
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def model_revision(self) -> str:
        return "test"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def encode(self, texts: Sequence[str]) -> NDArray[np.float64]:
        vectors = np.zeros((len(texts), self._dimensions), dtype=np.float64)
        for row, text in enumerate(texts):
            for token in _tokens(text):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                vectors[row, int.from_bytes(digest[:8], "big") % self._dimensions] += 1.0
        return vectors


class WrongWidthEncoder(HashingEncoder):
    """An encoder that returns a width the embedding column does not hold.

    The case the dimension assertion exists for: MedCPT's 768 is an expectation
    taken from its architecture and has not been confirmed against the weights.
    """

    def __init__(self, dimensions: int = 384) -> None:
        super().__init__(dimensions=dimensions, model_id="stub/wrong-width-encoder")


class OverlapCrossEncoder:
    """Scores a candidate by how many query words it contains."""

    @property
    def model_id(self) -> str:
        return "stub/overlap-cross-encoder"

    def score(self, query: str, texts: Sequence[str]) -> NDArray[np.float64]:
        wanted = set(_tokens(query))
        return np.array(
            [float(len(wanted.intersection(_tokens(text)))) for text in texts], dtype=np.float64
        )


@pytest.fixture
def encoder() -> HashingEncoder:
    """A deterministic dense encoder of the width the column holds."""
    return HashingEncoder()


@pytest.fixture
def cross_encoder() -> OverlapCrossEncoder:
    return OverlapCrossEncoder()


class HashingDocumentEncoder(HashingEncoder):
    """The pair-taking half of the stub pair.

    MedCPT's article encoder is fed two segments; this joins them before
    hashing, so a chunk whose section name differs embeds differently, which is
    the property the pair exists to create.
    """

    def __init__(
        self, dimensions: int = 768, model_id: str = "stub/hashing-document-encoder"
    ) -> None:
        super().__init__(dimensions=dimensions, model_id=model_id)

    def encode(self, pairs: Sequence[tuple[str, str]]) -> NDArray[np.float64]:  # type: ignore[override]
        return super().encode([" ".join(pair) for pair in pairs])


@pytest.fixture
def document_encoder() -> HashingDocumentEncoder:
    """A deterministic document encoder of the width the column holds."""
    return HashingDocumentEncoder()
