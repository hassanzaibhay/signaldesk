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
