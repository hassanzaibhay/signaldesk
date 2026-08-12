#!/usr/bin/env python3
"""Encoding and commit-metadata checks for tracked files.

Two rules, both mechanical:

1. Tracked text files are pure ASCII. Mixing ASCII hyphens with dashes, straight
   quotes with curly quotes, and plain spaces with non-breaking spaces produces
   diffs that are noisy to review and text that behaves differently depending on
   the console code page. A repository developed on Windows and built on Linux is
   exactly where those differences surface, so the rule is enforced rather than
   left to convention. Use "-" for a hyphen, "--" or a reworded sentence instead
   of a dash, straight quotes rather than curly ones, and "..." for an ellipsis.

2. No co-authorship trailers in tracked content. Commit metadata belongs in the
   commit, not in the working tree, and a stray trailer in a file confuses tools
   that parse trailers out of message bodies.

Data fixtures are exempt: real source data legitimately contains non-ASCII
characters, and rewriting it would corrupt the input the pipeline is measured on.

Usage:

    python scripts/check_hygiene.py            # every tracked file
    python scripts/check_hygiene.py PATH...    # only the given paths

Exits 0 when clean, 1 when any violation is found, 2 when it cannot inspect the
repository at all. Depends on the standard library only, so it runs on any host
with Python 3.12 and git, inside a container or out of one.
"""

from __future__ import annotations

import re
import subprocess
import sys
import unicodedata
from collections.abc import Iterable, Sequence
from pathlib import Path, PurePosixPath

ALLOWLIST: tuple[str, ...] = (
    "tests/fixtures/",
    "evals/golden/",
    "evals/reference_sets/",
    "evals/cassettes/",
    "data/overrides/",
)

TRAILER_PATTERN = re.compile(rb"^[ \t>#/*-]*co[-_ ]?authored?[-_ ]?by[ \t]*:", re.IGNORECASE)

MAX_REPORTED_PER_FILE = 10


class Violation:
    """A single rule breach, with enough location detail to fix it directly."""

    def __init__(self, path: str, line: int, column: int, detail: str) -> None:
        self.path = path
        self.line = line
        self.column = column
        self.detail = detail

    def render(self) -> str:
        return f"{self.path}:{self.line}:{self.column}: {self.detail}"


def run_git(args: Sequence[str]) -> str:
    """Run a read-only git command and return its stdout, or exit 2 if git fails."""
    try:
        completed = subprocess.run(
            ["git", *args],
            capture_output=True,
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        sys.stderr.write(f"cannot run git {' '.join(args)}: {exc}\n")
        raise SystemExit(2) from exc
    return completed.stdout


def repo_root() -> Path:
    return Path(run_git(["rev-parse", "--show-toplevel"]).strip())


def tracked_files() -> list[str]:
    output = run_git(["ls-files", "-z"])
    return [name for name in output.split("\0") if name]


def is_allowlisted(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in ALLOWLIST)


def binary_paths(names: Sequence[str]) -> set[str]:
    """Names git considers binary, per the binary attribute in .gitattributes."""
    if not names:
        return set()
    try:
        completed = subprocess.run(
            ["git", "check-attr", "--stdin", "-z", "binary"],
            input="\0".join(names) + "\0",
            capture_output=True,
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return set()
    fields = [field for field in completed.stdout.split("\0") if field]
    marked: set[str] = set()
    for index in range(0, len(fields) - 2, 3):
        if fields[index + 2] == "set":
            marked.add(fields[index])
    return marked


def check_bytes(name: str, data: bytes) -> list[Violation]:
    violations: list[Violation] = []
    for line_number, raw_line in enumerate(data.split(b"\n"), start=1):
        if TRAILER_PATTERN.search(raw_line):
            violations.append(
                Violation(name, line_number, 1, "co-authorship trailer in tracked content")
            )
        if all(byte < 0x80 for byte in raw_line):
            continue
        try:
            text = raw_line.decode("utf-8")
        except UnicodeDecodeError:
            violations.append(Violation(name, line_number, 1, "line is not valid UTF-8"))
            continue
        for column, character in enumerate(text, start=1):
            if character.isascii():
                continue
            codepoint = f"U+{ord(character):04X}"
            try:
                description = unicodedata.name(character)
            except ValueError:
                description = "unnamed codepoint"
            violations.append(
                Violation(name, line_number, column, f"non-ASCII {codepoint} {description}")
            )
    return violations


def check_files(root: Path, names: Iterable[str]) -> list[Violation]:
    candidates = [name for name in names if not is_allowlisted(name)]
    skip = binary_paths(candidates)
    violations: list[Violation] = []
    for name in candidates:
        if name in skip:
            continue
        path = root / PurePosixPath(name)
        if not path.is_file():
            continue
        data = path.read_bytes()
        if b"\0" in data:
            continue
        found = check_bytes(name, data)
        violations.extend(found[:MAX_REPORTED_PER_FILE])
        if len(found) > MAX_REPORTED_PER_FILE:
            remaining = len(found) - MAX_REPORTED_PER_FILE
            violations.append(Violation(name, 0, 0, f"...and {remaining} more in this file"))
    return violations


def normalize(root: Path, argument: str) -> str:
    """Turn a command-line path into a repository-relative POSIX name."""
    path = Path(argument)
    absolute = path if path.is_absolute() else Path.cwd() / path
    try:
        relative = absolute.resolve().relative_to(root.resolve())
    except ValueError:
        return argument.replace("\\", "/")
    return relative.as_posix()


def main(argv: Sequence[str]) -> int:
    root = repo_root()
    names = [normalize(root, argument) for argument in argv] if argv else tracked_files()
    violations = check_files(root, names)
    if not violations:
        sys.stdout.write(f"hygiene: {len(names)} files checked, no violations\n")
        return 0
    sys.stdout.write(f"hygiene: {len(violations)} violation(s)\n")
    for violation in violations:
        sys.stdout.write(f"  {violation.render()}\n")
    sys.stdout.write(
        "\nTracked files must be pure ASCII and free of co-authorship trailers.\n"
        "Replace dashes with '-' or '--', curly quotes with straight quotes, and\n"
        "ellipsis characters with '...'.\n"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
