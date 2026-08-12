"""Command line interface.

Every pipeline stage is a subcommand here, and the Makefile is a thin wrapper
around these commands. That way the documented interface and the executed
interface are the same thing.

Stages that later roadmap prompts implement raise ``NotImplementedError`` naming
the prompt that owns them, so an early clone fails with a useful message instead
of a silent no-op.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, NoReturn

import typer

from signaldesk import __version__
from signaldesk.core.logging import configure_logging, get_logger

log = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures"

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help=(
        "SignalDesk: post-market drug safety signal intelligence. "
        "Disproportionality statistics are hypothesis-generating and do not "
        "establish causation."
    ),
)

ingest_app = typer.Typer(
    no_args_is_help=True,
    help="Download and load source corpora. Idempotent and resumable; safe to re-run.",
)
normalize_app = typer.Typer(
    no_args_is_help=True,
    help="Map free-text source terms onto controlled vocabularies.",
)
signals_app = typer.Typer(
    no_args_is_help=True,
    help="Build contingency tables and disproportionality estimates.",
)
index_app = typer.Typer(
    no_args_is_help=True,
    help="Chunk, embed, and index the document corpora for retrieval.",
)
evals_app = typer.Typer(
    no_args_is_help=True,
    help="Run the evaluation suites and record their results.",
)
demo_app = typer.Typer(
    no_args_is_help=True,
    help="Load the committed fixture slice so the stack is usable after bootstrap.",
)

app.add_typer(ingest_app, name="ingest")
app.add_typer(normalize_app, name="normalize")
app.add_typer(signals_app, name="signals")
app.add_typer(index_app, name="index")
app.add_typer(evals_app, name="evals")
app.add_typer(demo_app, name="demo")


def _owned_by(prompt: str, stage: str) -> NoReturn:
    """Fail with the roadmap prompt that implements ``stage``."""
    message = f"{stage} is not implemented yet; roadmap prompt {prompt} implements it"
    raise NotImplementedError(message)


@app.callback()
def main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug-level logs.")] = False,
) -> None:
    """Configure logging before any subcommand runs."""
    configure_logging(debug=verbose)


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(__version__)


@ingest_app.command("faers")
def ingest_faers() -> None:
    """Download FAERS quarterly extracts, parse both schema eras, and load cases."""
    _owned_by("P02", "FAERS ingest")


@ingest_app.command("labels")
def ingest_labels() -> None:
    """Download openFDA prescription drug labels and split them into sections."""
    _owned_by("P05", "SPL label ingest")


@ingest_app.command("ctgov")
def ingest_ctgov() -> None:
    """Download ClinicalTrials.gov studies that have posted adverse event results."""
    _owned_by("P06", "ClinicalTrials.gov ingest")


@ingest_app.command("pubmed")
def ingest_pubmed() -> None:
    """Download the PubMed baseline and filter it to the adverse-effects subset."""
    _owned_by("P07", "PubMed ingest")


@normalize_app.command("drugs")
def normalize_drugs() -> None:
    """Map FAERS drug strings to RxNorm ingredients, recording method and score."""
    _owned_by("P03", "RxNorm normalization")


@signals_app.command("build")
def signals_build() -> None:
    """Recompute contingency tables and the four estimators over the corpus."""
    _owned_by("P08", "signal build")


@index_app.command("build")
def index_build() -> None:
    """Chunk documents, embed Tier A, and build the sparse and dense indexes."""
    _owned_by("P10", "index build")


@evals_app.command("run")
def evals_run(
    suite: Annotated[
        str,
        typer.Argument(
            help=(
                "Suite to run: retrieval, labeledness, briefs, refusal, causality, "
                "signals, ops, or all."
            )
        ),
    ],
) -> None:
    """Run an evaluation suite and append its metrics to the committed history."""
    _owned_by("P09 to P16", f"the {suite} evaluation suite")


@evals_app.command("record-cassettes")
def evals_record_cassettes() -> None:
    """Record model responses so continuous integration can replay them offline."""
    _owned_by("P11", "cassette recording")


@demo_app.command("load")
def demo_load(
    fixtures: Annotated[
        Path | None,
        typer.Option("--fixtures", help="Directory holding the committed fixture slice."),
    ] = None,
) -> None:
    """Load the committed fixture slice.

    Bootstrap calls this on a fresh stack, so it must succeed on a repository
    that has no fixtures yet: it reports that there is nothing to load and
    exits cleanly. The loaders themselves arrive with the ingest prompts.
    """
    directory = fixtures or DEFAULT_FIXTURE_DIR
    if not directory.is_dir():
        log.info("demo.load.no_fixtures", reason="directory_missing", path=str(directory))
        typer.echo(f"No fixture slice at {directory}; nothing to load.")
        return

    files = sorted(
        path for path in directory.rglob("*") if path.is_file() and path.name != ".gitkeep"
    )
    if not files:
        log.info("demo.load.no_fixtures", reason="directory_empty", path=str(directory))
        typer.echo(f"No fixture slice in {directory}; nothing to load.")
        return

    log.info("demo.load.pending", path=str(directory), files=len(files))
    typer.echo(f"Found {len(files)} fixture file(s) in {directory}; loaders arrive with P02.")


if __name__ == "__main__":
    app()
