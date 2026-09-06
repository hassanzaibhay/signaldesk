"""Batching, normalisation, and the dimension assertion.

The assertion is the reason this logic sits outside the model adapters. MedCPT's
768 is taken from its architecture and has not been checked against the weights,
so the guess may be wrong; what must not happen is a wrong guess producing an
index that is quietly not what it says it is.
"""

from __future__ import annotations

import numpy as np
import pytest
from tests.conftest import HashingEncoder, WrongWidthEncoder

from signaldesk.rag.embed import (
    EmbeddingError,
    assert_dimensions,
    batched,
    embed_texts,
    normalize_rows,
)

pytestmark = pytest.mark.unit

TEXTS = ["lactic acidosis", "hepatic failure", "rash and pruritus", "no known reaction"]


class TestBatching:
    def test_texts_are_split_into_batches_of_at_most_size(self) -> None:
        assert [list(batch) for batch in batched(TEXTS, 3)] == [TEXTS[:3], TEXTS[3:]]

    def test_a_batch_larger_than_the_input_yields_one_batch(self) -> None:
        assert [list(batch) for batch in batched(TEXTS, 99)] == [TEXTS]

    def test_a_batch_size_below_one_is_refused(self) -> None:
        """It would loop forever rather than fail."""
        with pytest.raises(ValueError, match="at least 1"):
            list(batched(TEXTS, 0))


class TestTheDimensionAssertion:
    def test_an_encoder_of_the_wrong_width_is_refused(self) -> None:
        """Not truncated, not padded. Either would make the index a liar."""
        with pytest.raises(EmbeddingError, match="384-dimensional"):
            embed_texts(WrongWidthEncoder(), TEXTS, expected_dimensions=768)

    def test_the_refusal_names_both_widths_and_the_fix(self) -> None:
        """ "Dimension mismatch" sends a reader to the source. This sends them to the decision."""
        with pytest.raises(EmbeddingError) as caught:
            embed_texts(WrongWidthEncoder(), TEXTS, expected_dimensions=768)

        message = str(caught.value)
        assert "384" in message
        assert "768" in message
        assert "EMBEDDING_DIMENSIONS" in message
        assert "migration" in message

    def test_it_fails_on_the_first_batch_rather_than_the_last(self) -> None:
        """A corpus-wide run must not do hours of work before noticing."""
        calls: list[int] = []

        class Counting(WrongWidthEncoder):
            def encode(self, texts):  # type: ignore[no-untyped-def]
                calls.append(len(texts))
                return super().encode(texts)

        with pytest.raises(EmbeddingError):
            embed_texts(Counting(), TEXTS, expected_dimensions=768, batch_size=1)

        assert calls == [1]

    def test_a_one_dimensional_return_is_refused(self) -> None:
        assert_dimensions(np.zeros((2, 4)), 4)
        with pytest.raises(EmbeddingError, match="two-dimensional"):
            assert_dimensions(np.zeros(4), 4)

    def test_a_row_count_that_does_not_match_the_texts_is_refused(self) -> None:
        """Rows and texts correspond by position; a mismatch mislabels every row."""

        class Dropping(HashingEncoder):
            def encode(self, texts):  # type: ignore[no-untyped-def]
                return super().encode(texts)[:-1]

        with pytest.raises(EmbeddingError, match="correspond by position"):
            embed_texts(Dropping(), TEXTS, expected_dimensions=768, batch_size=4)


class TestNormalisation:
    def test_rows_come_out_at_unit_length(self) -> None:
        unit = normalize_rows(np.array([[3.0, 4.0], [1.0, 0.0]]))

        assert np.allclose(np.linalg.norm(unit, axis=1), 1.0)

    def test_a_zero_row_stays_zero_rather_than_becoming_nan(self) -> None:
        """A NaN row poisons every distance computed against it."""
        unit = normalize_rows(np.array([[0.0, 0.0], [3.0, 4.0]]))

        assert np.array_equal(unit[0], np.zeros(2))
        assert not np.isnan(unit).any()

    def test_embedding_normalises_what_it_returns(self) -> None:
        vectors = embed_texts(HashingEncoder(), TEXTS, expected_dimensions=768)

        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


class TestEmbedding:
    def test_row_order_matches_text_order_across_batches(self) -> None:
        """A reordering here silently attaches every vector to the wrong chunk."""
        encoder = HashingEncoder()
        batched_rows = embed_texts(encoder, TEXTS, expected_dimensions=768, batch_size=1)
        whole = embed_texts(encoder, TEXTS, expected_dimensions=768, batch_size=99)

        assert np.allclose(batched_rows, whole)

    def test_no_texts_gives_an_empty_matrix_of_the_right_width(self) -> None:
        vectors = embed_texts(HashingEncoder(), [], expected_dimensions=768)

        assert vectors.shape == (0, 768)

    def test_texts_sharing_words_are_nearer_than_texts_that_do_not(self) -> None:
        """The stub has to rank meaningfully or it proves nothing downstream."""
        vectors = embed_texts(
            HashingEncoder(),
            ["lactic acidosis reported", "lactic acidosis observed", "hepatic failure"],
            expected_dimensions=768,
        )

        assert vectors[0] @ vectors[1] > vectors[0] @ vectors[2]
