"""Turning label sections into retrievable windows.

The outer boundary is the section and the inner boundary is the sentence. A
chunk never spans two sections, because a statement in a boxed warning and a
statement in an adverse reactions list are different strengths of claim about
the same drug, and the labeledness question this corpus exists to answer turns
on exactly that difference. Flattening the two would discard the most useful
structure the source has.

Inside a section, sentences are packed into windows of ``chunk_target_tokens``
with ``chunk_overlap_tokens`` carried between them. Both come from the settings
object and are not redefined here.

What this loses, stated where someone reading a disappointing retrieval result
will find it:

* Adverse-reaction sections are largely tabular term lists under body-system
  headings. A sentence splitter treats a long comma-delimited run as one unit
  or divides it arbitrarily, and a window boundary can separate a term from the
  heading and the frequency band that qualify it. A chunk can read "asthenia,
  fever" with nothing recording which body system or how often. The overlap
  reduces this and does not remove it.
* Cross-section reference is lost. A boxed warning that says "see Warnings and
  Precautions" does not carry its referent into the same chunk.

Token counts here are estimates, not tokenizer output. The tokenizer lives with
the model, the model is not a dependency of this module, and neither is
available in continuous integration. ``estimate_tokens`` therefore errs high, so
that a window sized against it stays inside the encoder's real limit rather than
being silently truncated at embed time. The embed step asserts the true length
against the model and refuses; see ``rag.embed``.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    import pysbd  # type: ignore[import-untyped]

#: Wordpiece tokens per whitespace-delimited word, rounded up rather than to the
#: nearest. Biomedical prose runs about 1.3 to 1.6 on a PubMedBERT vocabulary
#: because drug and reaction names fragment; 1.6 is the top of that range and is
#: used deliberately. Estimating low would build windows that the encoder then
#: truncates, dropping text that the index would still claim to contain.
TOKENS_PER_WORD: Final = 1.6

#: Characters per wordpiece, as a second floor. A run with few spaces - a long
#: hyphenated chemical name, a table flattened into one line - has fewer words
#: than its length implies, and the word-based estimate alone would understate
#: it. The estimate is the larger of the two.
CHARS_PER_TOKEN: Final = 3.5

#: The two special tokens every BERT-family encoder adds around a sequence.
SPECIAL_TOKENS: Final = 2

_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class Chunk:
    """One retrievable window of one section.

    ``sha256`` covers the section code as well as the text. The same paragraph
    appearing under ``warnings`` and under ``adverse_reactions`` is two
    different claims and must not collapse into one row on its way to the
    index, however identical the characters are.
    """

    text: str
    section_code: str
    #: Position within the section, from zero. Order is meaning in a label.
    ordinal: int
    #: Conservative wordpiece estimate; see the module docstring.
    token_estimate: int
    sha256: str


def normalize(text: str) -> str:
    """Collapse whitespace runs and strip.

    Label text arrives with the line wrapping of whatever produced the SPL, and
    that wrapping is not information. Normalising here means two labels that say
    the same thing hash the same, which is what makes deduplication find the 54
    percent of section rows that are byte-identical to another once the wrapping
    is set aside.
    """
    return _WHITESPACE.sub(" ", text).strip()


def _estimate_from_counts(words: int, characters: int) -> int:
    """The estimate, from counts rather than from text.

    Extracted so that the incremental packer in ``_split_oversized`` and
    ``estimate_tokens`` cannot drift apart. They did: the packer sized windows
    on the word bound alone, the character bound is the one that binds on a
    flattened term list, and windows came out at nearly twice the target.
    """
    if words == 0:
        return 0
    by_words = words * TOKENS_PER_WORD
    by_chars = characters / CHARS_PER_TOKEN
    return int(max(by_words, by_chars)) + SPECIAL_TOKENS


def estimate_tokens(text: str) -> int:
    """A deliberately high estimate of the wordpiece length of ``text``.

    High, not accurate. See the module docstring for why the direction of the
    error is the point.
    """
    stripped = text.strip()
    if not stripped:
        return 0
    return _estimate_from_counts(len(stripped.split()), len(stripped))


def chunk_sha256(section_code: str, text: str) -> str:
    """The identity of a chunk: what it says, and where it said it."""
    payload = section_code.encode("utf-8") + b"|" + text.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@lru_cache(maxsize=1)
def _segmenter() -> pysbd.Segmenter:
    """One segmenter, built once.

    Constructing a ``pysbd.Segmenter`` compiles a large body of regular
    expressions. Doing that once per section would dominate a corpus-wide run.
    """
    import pysbd

    return pysbd.Segmenter(language="en", clean=False)


def split_sentences(text: str) -> list[str]:
    """Sentences, normalised, with empties dropped."""
    normalized = normalize(text)
    if not normalized:
        return []
    pieces = (normalize(piece) for piece in _segmenter().segment(normalized))
    return [piece for piece in pieces if piece]


def _split_oversized(sentence: str, target_tokens: int) -> list[str]:
    """Break a sentence that is larger than a whole window.

    Not hypothetical: a single label section can run far longer than one
    window, and an adverse-reactions table flattened into prose can be one
    comma-run that no sentence splitter divides. Left alone such a sentence either becomes
    a chunk the encoder truncates or is dropped for not fitting. Both lose text
    while the index goes on claiming to hold it, so it is cut on word
    boundaries instead - a worse chunk than a sentence, and an honest one.
    """
    words = sentence.split()
    if not words:
        return []

    windows: list[str] = []
    start = 0
    count = 0
    characters = 0
    for position, word in enumerate(words):
        # The width this word would add, including the space that joins it.
        added = len(word) + (1 if count else 0)
        if count and _estimate_from_counts(count + 1, characters + added) > target_tokens:
            windows.append(" ".join(words[start:position]))
            start, count, characters = position, 1, len(word)
            continue
        count += 1
        characters += added
    windows.append(" ".join(words[start:]))
    return windows


def _overlap_tail(sentences: list[str], overlap_tokens: int) -> list[str]:
    """The trailing sentences of a window that begin the next one.

    Taken whole. A partial sentence carried across a boundary reads as a
    fragment in both windows and helps neither.
    """
    if overlap_tokens <= 0:
        return []
    tail: list[str] = []
    carried = 0
    for sentence in reversed(sentences):
        cost = estimate_tokens(sentence)
        if carried + cost > overlap_tokens:
            break
        tail.insert(0, sentence)
        carried += cost
    return tail


def chunk_section(
    text: str,
    section_code: str,
    *,
    target_tokens: int,
    overlap_tokens: int,
) -> tuple[Chunk, ...]:
    """Pack one section's text into overlapping windows.

    Greedy: sentences accumulate until the next one would not fit, the window is
    emitted, and the next window opens with whichever trailing sentences fit
    inside the overlap budget.
    """
    if overlap_tokens >= target_tokens:
        message = (
            f"overlap must be smaller than the window (got {overlap_tokens} >= {target_tokens}); "
            "an overlap at or above the target never advances and would not terminate"
        )
        raise ValueError(message)

    sentences: list[str] = []
    for sentence in split_sentences(text):
        if estimate_tokens(sentence) > target_tokens:
            sentences.extend(_split_oversized(sentence, target_tokens))
        else:
            sentences.append(sentence)

    windows: list[list[str]] = []
    current: list[str] = []
    budget = 0
    for sentence in sentences:
        cost = estimate_tokens(sentence)
        if current and budget + cost > target_tokens:
            windows.append(current)
            current = _overlap_tail(current, overlap_tokens)
            budget = sum(estimate_tokens(piece) for piece in current)
        current.append(sentence)
        budget += cost
    if current:
        windows.append(current)

    chunks = []
    for ordinal, window in enumerate(windows):
        body = " ".join(window)
        chunks.append(
            Chunk(
                text=body,
                section_code=section_code,
                ordinal=ordinal,
                token_estimate=estimate_tokens(body),
                sha256=chunk_sha256(section_code, body),
            )
        )
    return tuple(chunks)
