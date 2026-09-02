#!/usr/bin/env python3
"""Encoding, commit-metadata and artifact-tracking checks.

Three rules, all mechanical:

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

3. Nothing sits untracked under evals/history/ or evals/golden/. CLAUDE.md
   requires every published number to trace to a committed artifact there, and
   rules 1 and 2 read `git ls-files`, so an artifact that is never staged is
   never checked and never noticed. Six accumulated that way, including the run
   backing the figures quoted for this pipeline. An artifact deliberately not to
   be committed - a run that measured nothing, a warm re-run whose figures are
   wrong - gets deleted, which is what happened to two of them. It does not get
   left sitting stageable, because a directory a reader believes is the
   provenance record is worse when it is silently partial than when it is
   visibly wrong. evals/golden/ is covered by the same rule for a different
   reason: a gold set is hand-curated and cannot be regenerated, so an
   uncommitted one is not a missing number but lost work.

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

#: The interpreter this gate is defined against. Everything here is stdlib, so
#: the checks run happily on a Python the project does not support and report
#: success for a gate that never ran under the pinned interpreter. The version
#: is asserted rather than assumed, because a gate that cannot fail is not a
#: gate.
REQUIRED_PYTHON: tuple[int, int] = (3, 12)
REQUIRED_IMPLEMENTATION = "cpython"


class Violation:
    """A single rule breach, with enough location detail to fix it directly."""

    def __init__(self, path: str, line: int, column: int, detail: str) -> None:
        self.path = path
        self.line = line
        self.column = column
        self.detail = detail

    def render(self) -> str:
        return f"{self.path}:{self.line}:{self.column}: {self.detail}"


def check_interpreter(
    version: tuple[int, int] | None = None,
    implementation: str | None = None,
) -> str | None:
    """Return why the running interpreter is unacceptable, or None if it is fine.

    Arguments default to the live interpreter and exist so the rejection paths
    are testable without spawning a subprocess for each one.
    """
    version = sys.version_info[:2] if version is None else version
    implementation = sys.implementation.name if implementation is None else implementation
    if version == REQUIRED_PYTHON and implementation == REQUIRED_IMPLEMENTATION:
        return None
    required = f"{REQUIRED_IMPLEMENTATION} {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}"
    found = f"{implementation} {version[0]}.{version[1]}"
    return f"hygiene: requires {required}, found {found}"


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


#: The directories whose contents must all be committed.
#:
#: ``evals/history/`` is the provenance record every published number traces to.
#: ``evals/golden/`` holds the hand-curated gold sets, which are the one class of
#: artifact here that cannot be regenerated at all: a labeledness gold set is a
#: day of annotation, and the machine it was produced on has already been rebuilt
#: once mid-project. Untracked fires; tracked-and-modified does not, so the cost
#: of the rule is committing the file at the end of each annotation session,
#: which is the habit it is meant to enforce.
ARTIFACT_ROOTS = ("evals/history/", "evals/golden/")


def untracked_artifacts() -> list[str]:
    """Files under :data:`ARTIFACT_ROOTS` that git neither tracks nor ignores.

    ``--others`` lists untracked paths and ``--exclude-standard`` applies the
    ignore rules, so a path deliberately ignored is not reported. Nothing under
    these directories is ignored today, which is the point: an artifact written
    there is stageable, invisible to every other check here, and one ``git add``
    away from being published without ever having been inspected.
    """
    output = run_git(["ls-files", "-z", "--others", "--exclude-standard", "--", *ARTIFACT_ROOTS])
    return sorted(name for name in output.split("\0") if name)


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
    problem = check_interpreter()
    if problem is not None:
        sys.stdout.write(f"{problem}\n")
        return 2
    root = repo_root()
    explicit = bool(argv)
    names = [normalize(root, argument) for argument in argv] if explicit else tracked_files()
    violations = check_files(root, names)

    # Only on a whole-repository run. Given explicit paths the caller is asking
    # about those files, and failing on an unrelated untracked artifact would
    # make a targeted check unusable.
    stray = [] if explicit else untracked_artifacts()
    for name in stray:
        violations.append(Violation(name, 1, 1, "untracked artifact; commit it or delete it"))

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
    if stray:
        roots = " and ".join(ARTIFACT_ROOTS)
        sys.stdout.write(
            f"\nEvery file under {roots} must be committed. That is where published\n"
            "numbers are traced to and where the hand-curated gold sets live, and the\n"
            "other checks here only see tracked files. A run that measured nothing, or\n"
            "whose figures are known wrong, gets deleted rather than left untracked.\n"
            "A gold set is never deleted; it cannot be produced again.\n"
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
