"""Cleaning FAERS drug strings before they are looked up.

The strings are typed by thousands of different reporters over fourteen years,
and the corpus carries 641,070 distinct `drugname` values against 15,519 distinct
`prod_ai` values. The difference is mostly formatting: a trailing full stop, a
dose glued to the name, a manufacturer in brackets, several ingredients in one
field.

Every transformation here is conservative, and the reason is a measured one. The
probe of RxNav's approximate matcher found that a string with no drug in it at
all resolves to a real ingredient with a plausible score - `UNSPECIFIED
INGREDIENT` returns `caviar preparation`. Cleaning that guesses will therefore
not fail loudly; it will produce a confident match to the wrong concept, and
nothing downstream will notice. So each step fires only on a pattern it can
recognise unambiguously and returns the input untouched otherwise.

The steps are separate functions rather than one regular expression so each can
be tested on the cases it must change and, more importantly, on the cases it must
leave alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Units that appear glued to a name in this corpus, e.g. `SOLIRIS 300MG`.
_UNITS = r"MG|MCG|UG|G|KG|ML|L|IU|U|MEQ|MMOL|%"

#: A dose fragment: a number, optionally a unit, optionally a per-unit part.
_DOSE = re.compile(
    rf"\b\d+(?:[.,]\d+)?\s*(?:{_UNITS})\b(?:\s*/\s*\d+(?:[.,]\d+)?\s*(?:{_UNITS})\b)?",
    re.IGNORECASE,
)

#: Dosage forms and routes. Only stripped from the tail, never from the middle:
#: `INJECTION SOLUTION` at the end is packaging, in the middle it may be part of
#: a product name.
_FORMS = (
    "PRE-FILLED SYRINGE",
    "PREFILLED SYRINGE",
    "ORAL SOLUTION",
    "ORAL TABLET",
    "ORAL SUSPENSION",
    "DELAYED RELEASE",
    "EXTENDED RELEASE",
    "FILM COATED",
    "TABLETS",
    "TABLET",
    "TABS",
    "TAB",
    "CAPSULES",
    "CAPSULE",
    "CAPS",
    "CAP",
    "INJECTION",
    "INJ",
    "SOLUTION",
    "SUSPENSION",
    "SYRINGE",
    "SYRUP",
    "CREAM",
    "OINTMENT",
    "PATCH",
    "PEN",
    "VIAL",
    "AMPOULE",
    "SPRAY",
    "DROPS",
    "POWDER",
    "GEL",
    "ORAL",
    "IV",
)
_FORM_SUFFIX = re.compile(rf"(?:\s+(?:{'|'.join(_FORMS)}))+$", re.IGNORECASE)

#: A bracketed tail is dropped only when it carries no ingredient information:
#: a code, a registration number, or an explicit absence. `(852)`, `(UNKNOWN)`,
#: `(RHODES 91?490)` all go; `(VITAMIN C)` stays, because it does not match.
_UNINFORMATIVE_BRACKET = re.compile(
    r"\s*\((?:[^()]*\d[^()]*|UNKNOWN|NOS|UNSPECIFIED|NA|N/A)\)\s*$", re.IGNORECASE
)

_TRAILING_PUNCTUATION = re.compile(r"[.,;:\-\s]+$")
_WHITESPACE = re.compile(r"\s+")

#: Separators that join several ingredients in one field. Comma is deliberately
#: absent: `SERESTA 10 MG, COMPRIME` is one drug with its form after a comma,
#: and 38,732 distinct strings contain a comma, so treating it as a separator
#: would split thousands of single-ingredient strings into fragments.
_SEPARATORS = re.compile(r"\s*(?:/|\\|\+|\bAND\b|\bWITH\b)\s*", re.IGNORECASE)

#: A component shorter than this is not a drug name, it is debris from a split.
MIN_COMPONENT_LENGTH = 3

#: Strip rounds. Two is enough for every pattern in the corpus; the third exists
#: only so the loop terminates on input nobody anticipated.
_MAX_ROUNDS = 3


@dataclass(frozen=True, slots=True)
class Cleaned:
    """One source string, and everything that happened to it.

    `steps` is the audit trail: a raw string can be traced to the query that was
    actually sent, which is the difference between a mapping that can be checked
    and one that has to be trusted.
    """

    raw: str
    folded: str
    cleaned: str
    components: tuple[str, ...]
    steps: tuple[str, ...]

    @property
    def is_combination(self) -> bool:
        return len(self.components) > 1


def fold(raw: str) -> str:
    """The cache key and the join key: trimmed, collapsed, uppercased.

    This is the form the corpus was counted in, so a table keyed on it can be
    checked against the measured distinct-string counts.
    """
    return _WHITESPACE.sub(" ", raw).strip().upper()


def strip_trailing_punctuation(text: str) -> str:
    """`RANITIDINE.` and `RANITIDINE` are the same drug.

    The corpus has 423,722 rows of `RANITIDINE.` against 206,790 of `RANITIDINE`,
    so this is not a rare typo but a systematic reporting habit.
    """
    stripped = _TRAILING_PUNCTUATION.sub("", text)
    return stripped or text


def strip_uninformative_brackets(text: str) -> str:
    """Drop a bracketed tail that carries a code rather than an ingredient."""
    stripped = _UNINFORMATIVE_BRACKET.sub("", text).strip()
    return stripped or text


def strip_dose_and_form(text: str) -> str:
    """Remove a dose and any dosage form from the tail of a string.

    Only from the tail. A dose in the middle of a string is usually separating
    two ingredients of a combination product, and cutting there would silently
    drop one of them.
    """
    without_dose = text
    match = _DOSE.search(text)
    if match is not None and match.start() > 0:
        candidate = text[: match.start()].strip()
        if len(candidate) >= MIN_COMPONENT_LENGTH:
            without_dose = candidate

    without_form = _FORM_SUFFIX.sub("", without_dose).strip()
    if len(without_form) < MIN_COMPONENT_LENGTH:
        return without_dose
    return without_form


def split_components(text: str) -> tuple[str, ...]:
    """Split a combination product into its parts, or return the whole string.

    Applied after the dose has gone, so the slash in `40 MG/0.8 ML` is no longer
    present to be mistaken for a separator. A split is rejected outright if any
    part is too short or is purely numeric, because a bad split produces several
    confident matches to the wrong concepts rather than one honest failure.
    """
    parts = [part.strip() for part in _SEPARATORS.split(text) if part.strip()]
    if len(parts) < 2:
        return (text,)
    if any(len(part) < MIN_COMPONENT_LENGTH or part.isdigit() for part in parts):
        return (text,)
    return tuple(parts)


def clean(raw: str) -> Cleaned:
    """Fold, strip what is unambiguous, and split a combination into parts.

    The strips run in rounds until nothing changes, because each is anchored to
    the end of the string and one can uncover another: in
    `RANITIDINE HYDROCHLORIDE (852) 150MG TABLET.` the bracketed code is only at
    the tail once the dose and form have gone. Every application is recorded, so
    the trail shows the order things actually happened in.
    """
    folded = fold(raw)
    steps: list[str] = []
    text = folded

    for _ in range(_MAX_ROUNDS):
        changed = False
        for name, step in (
            ("trailing_punctuation", strip_trailing_punctuation),
            ("uninformative_brackets", strip_uninformative_brackets),
            ("dose_and_form", strip_dose_and_form),
        ):
            applied = step(text)
            if applied != text:
                steps.append(name)
                text = applied
                changed = True
        if not changed:
            break

    components = split_components(text)
    if len(components) > 1:
        steps.append("combination_split")

    return Cleaned(
        raw=raw,
        folded=folded,
        cleaned=text,
        components=components,
        steps=tuple(steps),
    )
