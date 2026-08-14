"""The quarter as a value object.

Quarters are the unit of everything here: the unit of download, of the manifest,
of the Parquet partition, and of the schema era. They arrive as strings like
"2014Q3" from the command line, from directory names, and from partition paths,
and they need to sort and to step forward. A string does none of that correctly:
"2013Q4" < "2014Q1" happens to hold lexically, but "2009Q4" < "2010Q1" only holds
because the years are the same width, and nothing stops "2014Q5".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Self

QUARTER_PATTERN = re.compile(r"^(?P<year>\d{4})[Qq](?P<quarter>[1-4])$")

MIN_YEAR = 2004
MAX_YEAR = 2100


@dataclass(frozen=True, order=True, slots=True)
class Quarter:
    """A calendar quarter, ordered by year then quarter."""

    year: int
    quarter: int

    def __post_init__(self) -> None:
        if not MIN_YEAR <= self.year <= MAX_YEAR:
            message = f"year out of range: {self.year}"
            raise ValueError(message)
        if not 1 <= self.quarter <= 4:
            message = f"quarter must be 1 to 4, got {self.quarter}"
            raise ValueError(message)

    @classmethod
    def parse(cls, text: str) -> Self:
        """Parse "2014Q3". The quarter letter may be either case."""
        match = QUARTER_PATTERN.match(text.strip())
        if match is None:
            message = f"not a quarter: {text!r}, expected a form like 2014Q3"
            raise ValueError(message)
        return cls(year=int(match["year"]), quarter=int(match["quarter"]))

    def __str__(self) -> str:
        return f"{self.year}Q{self.quarter}"

    @property
    def label(self) -> str:
        """The canonical rendering, used in paths and partition values."""
        return str(self)

    def next(self) -> Quarter:
        """The following quarter."""
        if self.quarter == 4:
            return Quarter(self.year + 1, 1)
        return Quarter(self.year, self.quarter + 1)

    def previous(self) -> Quarter:
        """The preceding quarter."""
        if self.quarter == 1:
            return Quarter(self.year - 1, 4)
        return Quarter(self.year, self.quarter - 1)

    @staticmethod
    def range(start: Quarter, end: Quarter) -> list[Quarter]:
        """Every quarter from start to end inclusive, ascending.

        An end before the start is an empty range rather than an error: callers
        pass user input straight in, and "nothing to do" is the honest answer.
        """
        if end < start:
            return []
        quarters = [start]
        while quarters[-1] < end:
            quarters.append(quarters[-1].next())
        return quarters
