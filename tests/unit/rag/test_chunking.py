"""Section-aware chunking: boundaries, overlap, and the cases that lose text.

No model and no database. The token counts here are the module's own
conservative estimate, which is the number the packing decision actually uses;
asserting against a real tokenizer would test a dependency that continuous
integration does not have.
"""

from __future__ import annotations

import pytest

from signaldesk.rag.chunking import (
    SPECIAL_TOKENS,
    Chunk,
    chunk_section,
    chunk_sha256,
    estimate_tokens,
    normalize,
    split_sentences,
)

pytestmark = pytest.mark.unit


def _chunks(
    text: str, *, target: int = 40, overlap: int = 10, code: str = "warnings"
) -> tuple[Chunk, ...]:
    return chunk_section(text, code, target_tokens=target, overlap_tokens=overlap)


class TestNormalisation:
    def test_whitespace_runs_collapse(self) -> None:
        """SPL line wrapping is formatting, not information."""
        assert normalize("a  b\n\tc   \n d ") == "a b c d"

    def test_normalisation_is_what_makes_duplicates_hash_alike(self) -> None:
        """Two labels wrapped differently are the same claim."""
        one = chunk_sha256("warnings", normalize("Rash and\nfever."))
        two = chunk_sha256("warnings", normalize("Rash   and fever."))

        assert one == two

    def test_empty_text_has_no_sentences(self) -> None:
        assert split_sentences("   \n  ") == []


class TestTokenEstimates:
    def test_the_estimate_errs_high_rather_than_low(self) -> None:
        """A low estimate builds windows the encoder silently truncates."""
        text = "Hepatic failure and jaundice have been reported."
        words = len(text.split())

        assert estimate_tokens(text) > words

    def test_a_run_with_few_spaces_is_not_understated(self) -> None:
        """The character floor catches what the word count misses."""
        dense = "a" * 700

        assert estimate_tokens(dense) >= 200

    def test_empty_text_costs_nothing(self) -> None:
        assert estimate_tokens("   ") == 0

    def test_special_tokens_are_counted(self) -> None:
        assert estimate_tokens("word") >= SPECIAL_TOKENS


class TestIdentity:
    def test_the_same_text_under_two_sections_is_two_chunks(self) -> None:
        """A boxed warning and an adverse reactions list are different claims."""
        text = "Lactic acidosis has been reported."

        assert chunk_sha256("boxed_warning", text) != chunk_sha256("adverse_reactions", text)

    def test_identical_text_in_one_section_is_one_chunk(self) -> None:
        text = "Lactic acidosis has been reported."

        assert chunk_sha256("warnings", text) == chunk_sha256("warnings", text)


class TestPacking:
    def test_a_short_section_is_one_chunk(self) -> None:
        chunks = _chunks("Rash has been reported. Fever has been reported.")

        assert len(chunks) == 1
        assert chunks[0].ordinal == 0
        assert chunks[0].section_code == "warnings"

    def test_a_long_section_is_split_into_ordered_windows(self) -> None:
        text = " ".join(f"Finding number {n} was observed in the trial." for n in range(40))
        chunks = _chunks(text)

        assert len(chunks) > 1
        assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))

    def test_every_window_stays_within_the_target(self) -> None:
        """The whole point of the estimate is that windows fit the encoder."""
        text = " ".join(f"Finding number {n} was observed in the trial." for n in range(60))

        for chunk in _chunks(text, target=60, overlap=15):
            assert chunk.token_estimate <= 60

    def test_consecutive_windows_overlap(self) -> None:
        """The carried tail is what keeps a term near the sentence qualifying it."""
        text = " ".join(f"Sentence {n} describes a reaction." for n in range(30))
        chunks = _chunks(text, target=40, overlap=20)

        first_words = set(chunks[0].text.split())
        second_words = set(chunks[1].text.split())

        assert first_words & second_words

    def test_no_overlap_is_honoured(self) -> None:
        text = " ".join(f"Sentence {n} describes a reaction." for n in range(30))
        chunks = _chunks(text, target=40, overlap=0)
        joined = " ".join(chunk.text for chunk in chunks)

        # With no carried tail every sentence appears exactly once.
        assert joined.count("Sentence 0 ") == 1

    def test_an_overlap_at_or_above_the_target_is_refused(self) -> None:
        """It would never advance, so the packer would not terminate."""
        with pytest.raises(ValueError, match="never advances"):
            _chunks("Anything at all.", target=20, overlap=20)

    def test_empty_and_whitespace_sections_produce_nothing(self) -> None:
        assert _chunks("") == ()
        assert _chunks("   \n  ") == ()


class TestOversizedSentences:
    def test_a_sentence_larger_than_a_window_is_cut_rather_than_dropped(self) -> None:
        """The corpus really contains these.

        A flattened adverse-reactions table is one comma-run that no splitter
        divides. Dropping it would lose text while the index went on claiming to
        hold it.
        """
        run = ", ".join(f"reaction{n}" for n in range(400))
        chunks = _chunks(run, target=40, overlap=0)

        assert len(chunks) > 1
        assert "reaction0" in chunks[0].text
        assert "reaction399" in chunks[-1].text

    def test_cutting_an_oversized_sentence_keeps_windows_in_budget(self) -> None:
        run = ", ".join(f"reaction{n}" for n in range(400))

        for chunk in _chunks(run, target=40, overlap=0):
            assert chunk.token_estimate <= 40

    def test_no_text_is_lost_when_a_long_sentence_is_cut(self) -> None:
        run = " ".join(f"term{n}" for n in range(300))
        chunks = _chunks(run, target=40, overlap=0)
        recovered = " ".join(chunk.text for chunk in chunks).split()

        assert recovered == run.split()
