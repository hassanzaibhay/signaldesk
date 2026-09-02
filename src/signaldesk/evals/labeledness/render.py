"""Turning a screen into text, and nothing else.

Two properties this module has to hold, both of them testable.

**Nothing that correlates with the answer is rendered.** No disproportionality
statistic, no lexical stratum, no verdict tally, and no indication that a screen
is a repeat. :func:`render_screen` takes a :class:`Screen` and reads
``repeat_of`` nowhere; the test clears the field and asserts the bytes do not
move.

**No pre-highlighting.** Fewer than 3 percent of pairs in this frame have the
Preferred Term verbatim in the adverse-reactions text, so a pre-computed
highlight would turn its own absence into evidence of absence. Search is offered
and the annotator drives it. The terms the guideline requires are listed on the
screen, so the protocol is in front of the annotator rather than in their memory,
but running them is a keypress and not a default.
"""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass, replace
from typing import Final

from signaldesk.evals.labeledness.manifest import Protocol, Screen

WIDTH: Final = 96
PAGE_LINES: Final = 28

#: MedDRA filler that carries no search value. Guideline section 5.
FILLER_WORDS: Final[frozenset[str]] = frozenset(
    {"nos", "not", "otherwise", "specified", "disorder", "disorders"}
)
MIN_CONTENT_WORD: Final = 4

_SUFFIXES: Final[tuple[str, ...]] = ("iation", "ation", "ities", "ity", "ing", "ed", "es", "s")

SECTION_TITLES: Final[dict[str, str]] = {
    "adverse_reactions": "ADVERSE REACTIONS",
    "boxed_warning": "BOXED WARNING",
    "warnings_and_cautions": "WARNINGS AND PRECAUTIONS",
    "warnings": "WARNINGS",
}


def content_words(pt: str) -> tuple[str, ...]:
    """The Preferred Term's content words, per guideline section 5."""
    words = re.findall(r"[a-z]+", pt.lower())
    return tuple(
        word for word in words if len(word) >= MIN_CONTENT_WORD and word not in FILLER_WORDS
    )


def stem(word: str) -> str:
    """A deliberately crude suffix strip, published so a verdict is reproducible.

    Not a linguistic stemmer. It exists so that searching "elevat" reaches
    "elevations", and it is written out here rather than pulled from a library so
    that the exact string a search ran on is recoverable from this file alone.
    """
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= MIN_CONTENT_WORD:
            return word[: -len(suffix)]
    return word


def search_terms(pt: str) -> tuple[str, ...]:
    """The terms guideline section 5 requires, verbatim first then stems."""
    terms = [pt.lower()]
    for word in content_words(pt):
        rooted = stem(word)
        if rooted not in terms:
            terms.append(rooted)
    return tuple(terms)


def find_matches(text: str, term: str) -> tuple[int, ...]:
    """Every case-insensitive offset of ``term`` in ``text``."""
    if not term:
        return ()
    haystack = text.lower()
    needle = term.lower()
    found: list[int] = []
    start = haystack.find(needle)
    while start != -1:
        found.append(start)
        start = haystack.find(needle, start + 1)
    return tuple(found)


@dataclass(frozen=True, slots=True)
class ScreenView:
    """Which section is open, how far down it is, and the live search."""

    section_index: int = 0
    page: int = 0
    query: str = ""
    match_index: int = 0

    def with_section(self, index: int) -> ScreenView:
        return replace(self, section_index=index, page=0, match_index=0)

    def with_page(self, page: int) -> ScreenView:
        return replace(self, page=max(0, page))

    def with_query(self, query: str) -> ScreenView:
        return replace(self, query=query, match_index=0)


def _blocks_for(screen: Screen, section_code: str) -> tuple[str, ...]:
    return tuple(block.text for block in screen.sections if block.section_code == section_code)


def section_text(screen: Screen, section_code: str) -> str:
    """One section's blocks joined for display, published order preserved."""
    return "\n\n".join(_blocks_for(screen, section_code))


def _wrap(text: str) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            lines.append("")
            continue
        lines.extend(textwrap.wrap(paragraph, width=WIDTH) or [""])
    return lines


def _line_of_offset(text: str, offset: int) -> int:
    """Which wrapped line a character offset lands on, for search jumps."""
    prefix = text[:offset]
    consumed = 0
    for index, line in enumerate(_wrap(text)):
        consumed += len(line) + 1
        if consumed >= len(prefix):
            return index
    return max(0, len(_wrap(text)) - 1)


def _header(screen: Screen, progress: str) -> list[str]:
    present = " ".join(
        code if code in screen.section_codes else f"({code})"
        for code in ("adverse_reactions", "boxed_warning", "warnings_and_cautions", "warnings")
    )
    protocol = (
        "FULL: read the section, then search"
        if screen.protocol is Protocol.FULL
        else f"BOUNDED: search only ({screen.primary_chars} chars); an n here is a defensible n"
    )
    return [
        "=" * WIDTH,
        f"  DRUG   {screen.drug_string}",
        f"  EVENT  {screen.pt}",
        f"  LABEL  set_id {screen.set_id}",
        f"  READ   {protocol}",
        f"  HAVE   {present}",
        "=" * WIDTH,
        f"  {progress}",
        "-" * WIDTH,
    ]


def _footer(screen: Screen, view: ScreenView, matches: tuple[int, ...]) -> list[str]:
    terms = "  ".join(search_terms(screen.pt))
    if view.query:
        position = f"{view.match_index + 1}/{len(matches)}" if matches else "no match"
        # "." and "," rather than n/N: n is a verdict key and must never be
        # overloaded onto navigation, or a miskey writes a label.
        finding = f'  FIND "{view.query}"  {position}   . next   , prev'
    else:
        finding = "  /  search"
    return [
        "-" * WIDTH,
        f"  TERMS  {terms}",
        finding,
        "  1 adverse reactions   2 boxed   3 warnings+cautions   4 warnings",
        "  SPACE page down   B page up   G top",
        "  l explicit   b broader   c class   n not-labelled   u unclear",
        "  z undo the last verdict this session   q save and quit   ? guideline",
        "=" * WIDTH,
    ]


def visible_sections(screen: Screen) -> tuple[str, ...]:
    """The sections this document actually has, in guideline search order."""
    order = ("adverse_reactions", "boxed_warning", "warnings_and_cautions", "warnings")
    return tuple(code for code in order if code in screen.section_codes)


def render_screen(screen: Screen, *, progress: str, view: ScreenView) -> str:
    """The full frame for one screen.

    Pure in ``(screen, progress, view)``. It reads ``drug_string``, ``pt``,
    ``set_id``, ``protocol``, ``primary_chars`` and ``sections`` and nothing else
    off the screen, which is what keeps a repeat indistinguishable from its
    original.
    """
    sections = visible_sections(screen)
    if not sections:  # pragma: no cover - excluded by the frame
        return "\n".join([*_header(screen, progress), "  (no section text)", ""])
    index = min(view.section_index, len(sections) - 1)
    code = sections[index]
    text = section_text(screen, code)
    lines = _wrap(text)

    matches = find_matches(text, view.query) if view.query else ()
    page = view.page
    if matches:
        target = matches[view.match_index % len(matches)]
        page = _line_of_offset(text, target) // PAGE_LINES

    start = page * PAGE_LINES
    body = lines[start : start + PAGE_LINES]
    total_pages = max(1, (len(lines) + PAGE_LINES - 1) // PAGE_LINES)
    title = SECTION_TITLES.get(code, code.upper())

    frame = [
        *_header(screen, progress),
        f"  {title}   page {min(page + 1, total_pages)}/{total_pages}",
        "",
        *(f"  {line}" for line in body),
        *["" for _ in range(max(0, PAGE_LINES - len(body)))],
        *_footer(screen, view, matches),
    ]
    return "\n".join(frame) + "\n"


def render_note_prompt() -> str:
    """Shown after `u` only. The one place typing is worth the seconds."""
    return "  unclear: why? (guideline section 7; Enter to skip)\n  > "
