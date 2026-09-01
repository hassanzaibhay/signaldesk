"""The hygiene gate's interpreter guard.

The script lives in `scripts/`, not in the installed package, so it is loaded by
path rather than imported by name.

What is checked here is the guard, not the ASCII and trailer rules it protects.
Those rules are stdlib-only and run on any Python 3, which is the point: without
this guard the gate prints "no violations" and exits 0 under an interpreter the
project does not support, and a gate that cannot fail is not a gate.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.unit

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_hygiene.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_hygiene", SCRIPT)
    if spec is None or spec.loader is None:  # pragma: no cover - the file is tracked
        message = f"cannot load {SCRIPT}"
        raise RuntimeError(message)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def hygiene() -> ModuleType:
    return _load()


def test_the_pinned_interpreter_is_accepted(hygiene: ModuleType) -> None:
    assert hygiene.check_interpreter(version=(3, 12), implementation="cpython") is None


def test_the_suite_itself_runs_on_the_pinned_interpreter(hygiene: ModuleType) -> None:
    """The guard would be vacuous if the pin did not match what CI actually uses."""
    assert hygiene.check_interpreter() is None
    assert sys.version_info[:2] == hygiene.REQUIRED_PYTHON


@pytest.mark.parametrize(
    ("version", "implementation", "expected"),
    [
        ((3, 11), "cpython", "cpython 3.11"),
        ((3, 13), "cpython", "cpython 3.13"),
        ((3, 12), "pypy", "pypy 3.12"),
        ((3, 10), "pypy", "pypy 3.10"),
    ],
)
def test_everything_else_is_rejected_and_named(
    hygiene: ModuleType,
    version: tuple[int, int],
    implementation: str,
    expected: str,
) -> None:
    """A newer Python is rejected too: the pin is exact, not a floor."""
    problem = hygiene.check_interpreter(version=version, implementation=implementation)
    assert problem is not None
    assert expected in problem
    assert "cpython 3.12" in problem


def test_the_guard_short_circuits_main(
    hygiene: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Exit 2, and no file is read.

    2 is already this script's "could not run" code, which `run_git` uses when
    git is unavailable. 1 stays reserved for violations found, so a caller can
    still tell a broken gate from a failing one. `tracked_files` is replaced
    with a raiser to prove the guard returns before any work happens rather than
    merely printing alongside it.
    """

    def _fail() -> list[str]:  # pragma: no cover - reached only if the guard leaks
        message = "the guard did not short-circuit"
        raise AssertionError(message)

    monkeypatch.setattr(hygiene, "check_interpreter", lambda: "hygiene: nope")
    monkeypatch.setattr(hygiene, "tracked_files", _fail)

    assert hygiene.main([]) == 2
    assert capsys.readouterr().out == "hygiene: nope\n"


class TestNothingSitsUntrackedUnderEvalsHistory:
    """The third rule, and the reason it has to be a rule.

    Rules 1 and 2 read `git ls-files`, so an artifact that is written and never
    staged is invisible to them. Six accumulated that way before anyone looked,
    including the signal run the label pipeline scoped from and the run whose
    cached bytes the clean run replayed. "Remember to commit the artifact" is a
    human remembering, which is the shape this rule replaces.
    """

    def test_an_untracked_artifact_fails_the_whole_repository_run(
        self,
        hygiene: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # .git is not mounted into the container, so repo_root cannot run there.
        monkeypatch.setattr(hygiene, "repo_root", lambda: Path("/repo"))
        monkeypatch.setattr(hygiene, "tracked_files", list)
        monkeypatch.setattr(
            hygiene,
            "untracked_artifacts",
            lambda: ["evals/history/spl_ingest_20260901T114909Z.json"],
        )

        assert hygiene.main([]) == 1
        out = capsys.readouterr().out
        assert "spl_ingest_20260901T114909Z.json" in out
        assert "commit it or delete it" in out

    def test_a_clean_history_directory_passes(
        self,
        hygiene: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(hygiene, "repo_root", lambda: Path("/repo"))
        monkeypatch.setattr(hygiene, "tracked_files", list)
        monkeypatch.setattr(hygiene, "untracked_artifacts", list)

        assert hygiene.main([]) == 0
        assert "no violations" in capsys.readouterr().out

    def test_a_targeted_run_does_not_fail_on_an_unrelated_stray(
        self,
        hygiene: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Given explicit paths the caller is asking about those files.

        Failing a targeted check on an untracked artifact elsewhere would make
        it useless for the thing it is mostly used for, which is checking one
        file before staging it.
        """

        def _fail() -> list[str]:  # pragma: no cover - reached only on a scope leak
            message = "untracked_artifacts must not run for a targeted check"
            raise AssertionError(message)

        monkeypatch.setattr(hygiene, "repo_root", lambda: tmp_path)
        monkeypatch.setattr(hygiene, "untracked_artifacts", _fail)
        target = tmp_path / "plain.txt"
        target.write_text("ascii only\n", encoding="utf-8")

        assert hygiene.main([str(target)]) == 0

    def test_the_query_asks_git_for_untracked_and_unignored_paths(
        self, hygiene: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The flags are the rule.

        Without --others it reports nothing; without --exclude-standard it would
        report ignored files too and the gate would fail on data nobody intends
        to commit. Pinning the arguments keeps both halves honest.
        """
        seen: list[list[str]] = []

        def _capture(args: list[str]) -> str:
            seen.append(args)
            return "evals/history/stray.json\0"

        monkeypatch.setattr(hygiene, "run_git", _capture)

        assert hygiene.untracked_artifacts() == ["evals/history/stray.json"]
        assert seen == [
            ["ls-files", "-z", "--others", "--exclude-standard", "--", "evals/history/"]
        ]
