"""The CLI is the documented interface, so its shape is tested.

Unimplemented stages must fail loudly and say which roadmap prompt owns them;
``demo load`` must succeed on a repository that has no fixture slice yet,
because bootstrap runs it on a fresh clone.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import typer.main
from typer.core import TyperGroup
from typer.testing import CliRunner

from signaldesk import __version__
from signaldesk.cli import app

pytestmark = pytest.mark.unit

runner = CliRunner()


def test_help_lists_every_command_group() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for group in ("ingest", "normalize", "signals", "index", "evals", "demo"):
        assert group in result.stdout


def test_version_prints_the_package_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


@pytest.mark.parametrize(
    ("command", "prompt"),
    [
        (["ingest", "ctgov"], "P06"),
        (["ingest", "pubmed"], "P07"),
        (["normalize", "drugs"], "P03"),
        (["index", "build"], "P10"),
        (["evals", "record-cassettes"], "P11"),
    ],
)
def test_unimplemented_stage_names_its_roadmap_prompt(command: list[str], prompt: str) -> None:
    result = runner.invoke(app, command)
    assert result.exit_code != 0
    assert isinstance(result.exception, NotImplementedError)
    assert prompt in str(result.exception)


def test_implemented_signal_commands_are_listed() -> None:
    """The signal build is implemented, so it must no longer advertise a prompt."""
    result = runner.invoke(app, ["signals", "--help"])
    assert result.exit_code == 0
    for command in ("build", "artifact"):
        assert command in result.stdout


def test_implemented_ingest_commands_are_listed() -> None:
    """FAERS and SPL ingest are implemented, so they no longer advertise a prompt."""
    result = runner.invoke(app, ["ingest", "--help"])
    assert result.exit_code == 0
    for command in ("faers", "faers-dedup", "faers-status", "faers-quality", "labels"):
        assert command in result.stdout


def test_label_ingest_takes_a_scope_cap() -> None:
    """The first run is capped and measured rather than fetching every flagged
    string, so the cap has to be reachable from the command line.

    Read off the registered parameters rather than the rendered help. Typer
    renders help through Rich, which wraps to the terminal width and emits ANSI
    escapes, so a substring assertion against that text passes on a wide
    terminal and fails on a narrow one.
    """
    group = typer.main.get_command(app)
    assert isinstance(group, TyperGroup)
    ingest = group.commands["ingest"]
    assert isinstance(ingest, TyperGroup)
    options = {opt for param in ingest.commands["labels"].params for opt in param.opts}
    assert {"--top-k", "--run-id", "--force"} <= options


def test_evals_run_reports_the_suite_it_cannot_run_yet() -> None:
    result = runner.invoke(app, ["evals", "run", "retrieval"])
    assert isinstance(result.exception, NotImplementedError)
    assert "retrieval" in str(result.exception)


def test_demo_load_succeeds_when_the_fixture_directory_is_missing(tmp_path: Path) -> None:
    result = runner.invoke(app, ["demo", "load", "--fixtures", str(tmp_path / "absent")])
    assert result.exit_code == 0
    assert "nothing to load" in result.stdout


def test_demo_load_succeeds_when_only_a_placeholder_is_present(tmp_path: Path) -> None:
    (tmp_path / ".gitkeep").touch()
    result = runner.invoke(app, ["demo", "load", "--fixtures", str(tmp_path)])
    assert result.exit_code == 0
    assert "nothing to load" in result.stdout


def test_demo_load_counts_the_fixture_files_it_found(tmp_path: Path) -> None:
    (tmp_path / "cases.csv").write_text("primaryid\n1\n", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "reactions.csv").write_text("pt\nnausea\n", encoding="utf-8")

    result = runner.invoke(app, ["demo", "load", "--fixtures", str(tmp_path)])
    assert result.exit_code == 0
    assert "Found 2 fixture file(s)" in result.stdout
