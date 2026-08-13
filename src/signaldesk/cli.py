"""Command line interface.

Every pipeline stage is a subcommand here, and the Makefile is a thin wrapper
around these commands. That way the documented interface and the executed
interface are the same thing.

Stages that later roadmap prompts implement raise ``NotImplementedError`` naming
the prompt that owns them, so an early clone fails with a useful message instead
of a silent no-op.
"""

from __future__ import annotations

import os
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


def _setup_django() -> None:
    """Make the ORM usable from the command line.

    Commands that touch the database need the app registry loaded. Doing it here
    rather than at import time keeps `--help` fast and keeps the modules that do
    not need a database free of the dependency.
    """
    import django

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "signaldesk.web.config.settings.dev")
    django.setup()


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
def ingest_faers(
    from_quarter: Annotated[
        str, typer.Option("--from", help="First quarter, for example 2012Q4.")
    ] = "2012Q4",
    to_quarter: Annotated[
        str | None,
        typer.Option("--to", help="Last quarter. Defaults to the newest one published."),
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Re-ingest quarters already recorded as complete.")
    ] = False,
    keep_raw: Annotated[
        bool,
        typer.Option("--keep-raw", help="Leave downloaded files on disk. One quarter only."),
    ] = False,
    dedup: Annotated[
        bool, typer.Option("--dedup/--no-dedup", help="Run corpus-wide deduplication afterwards.")
    ] = True,
) -> None:
    """Download quarterly extracts, normalize them, and load Parquet and Postgres.

    Processes one quarter at a time and deletes its raw files before fetching the
    next, so peak disk stays at roughly one quarter regardless of range length.
    Re-running is a no-op for quarters already completed.
    """
    _setup_django()
    from signaldesk.ingest.faers import quality
    from signaldesk.ingest.faers.dedup import persist, resolve_versions_across_quarters, stage2
    from signaldesk.ingest.faers.discover import discover
    from signaldesk.ingest.faers.pipeline import ingest_quarter
    from signaldesk.ingest.faers.quarter import Quarter

    if keep_raw:
        log.warning(
            "faers.keep_raw.enabled",
            reason="raw files will not be deleted; use this for one quarter only",
        )
        typer.echo(
            "WARNING: --keep-raw retains every downloaded quarter. "
            "A full range this way needs well over a terabyte.",
            err=True,
        )

    discovery = discover()
    start = Quarter.parse(from_quarter)
    end = Quarter.parse(to_quarter) if to_quarter else discovery.latest
    quarters = Quarter.range(start, end)
    typer.echo(f"{len(quarters)} quarter(s): {start} to {end}")

    results = []
    for quarter in quarters:
        results.append(ingest_quarter(quarter, discovery, force=force, keep_raw=keep_raw))

    stats = None
    version_pairs: list[tuple[int, int]] = []
    if dedup:
        version_pairs = resolve_versions_across_quarters()
        stats = stage2()
        persist(stats, version_pairs)

    report = quality.build_report(results, stats, version_pairs=len(version_pairs))
    path = quality.write_report(report)
    typer.echo(quality.render(report))
    typer.echo(f"\nreport written to {path}")


@ingest_app.command("faers-dedup")
def ingest_faers_dedup(
    from_quarter: Annotated[str | None, typer.Option("--from", help="First quarter.")] = None,
    to_quarter: Annotated[str | None, typer.Option("--to", help="Last quarter.")] = None,
) -> None:
    """Deduplicate the loaded corpus.

    Separate from ingest because it is inherently corpus-wide: the same case is
    routinely re-reported under a different identifier in a later quarter, and a
    per-quarter pass cannot see those pairs. Re-running rebuilds the duplicate
    table rather than adding to it.
    """
    _setup_django()
    from signaldesk.ingest.faers.dedup import persist, resolve_versions_across_quarters, stage2
    from signaldesk.ingest.faers.quarter import Quarter

    start = Quarter.parse(from_quarter) if from_quarter else None
    end = Quarter.parse(to_quarter) if to_quarter else None

    version_pairs = resolve_versions_across_quarters()
    stats = stage2(start=start, end=end)
    rows = persist(stats, version_pairs)

    typer.echo(f"records considered:        {stats.records_considered:,}")
    typer.echo(
        f"excluded, null block key:  {stats.records_excluded_null_key:,} ({stats.excluded_share:.2%})"
    )
    typer.echo(f"blocks:                    {stats.blocks:,} (largest {stats.largest_block:,})")
    typer.echo(
        f"comparisons:               {stats.comparisons:,} (naive would be {stats.naive_comparisons:,})"
    )
    typer.echo(f"duplicate pairs:           {stats.duplicate_pairs:,}")
    typer.echo(f"cross-quarter share:       {stats.cross_quarter_share:.2%}")
    typer.echo(f"superseded across quarters:{len(version_pairs):,}")
    typer.echo(f"rows written:              {rows:,}")


@ingest_app.command("faers-status")
def ingest_faers_status() -> None:
    """Print the ingest manifest: what is loaded, what failed, what is missing."""
    _setup_django()
    from signaldesk.web.signals.models import IngestManifest

    rows = IngestManifest.objects.filter(source="faers").order_by("unit")
    if not rows:
        typer.echo("nothing ingested yet")
        return

    typer.echo(f"{'quarter':<9} {'status':<10} {'cases':>10} {'rows':>12} {'MB':>7}  checksum")
    for row in rows:
        cases = row.row_counts.get("case", 0) if isinstance(row.row_counts, dict) else 0
        typer.echo(
            f"{row.unit:<9} {row.status:<10} {cases:>10,} {row.row_count:>12,} "
            f"{row.bytes_downloaded / 1e6:>7.1f}  {row.checksum[:12]}"
        )
    typer.echo(f"\n{rows.count()} quarter(s) recorded")


@ingest_app.command("faers-quality")
def ingest_faers_quality(
    from_quarter: Annotated[str | None, typer.Option("--from", help="First quarter.")] = None,
    to_quarter: Annotated[str | None, typer.Option("--to", help="Last quarter.")] = None,
) -> None:
    """Recompute and print the quality report from what is loaded."""
    _setup_django()
    from signaldesk.ingest.faers import quality
    from signaldesk.ingest.faers.dedup import resolve_versions_across_quarters, stage2
    from signaldesk.ingest.faers.pipeline import QuarterResult
    from signaldesk.ingest.faers.quarter import Quarter
    from signaldesk.web.signals.models import IngestManifest

    start = Quarter.parse(from_quarter) if from_quarter else None
    end = Quarter.parse(to_quarter) if to_quarter else None

    results = []
    for row in IngestManifest.objects.filter(source="faers", status="completed").order_by("unit"):
        quarter = Quarter.parse(row.unit)
        if (start and quarter < start) or (end and quarter > end):
            continue
        result = QuarterResult(quarter=quarter)
        result.row_counts = dict(row.row_counts or {})
        result.had_deleted_file = row.had_deleted_file
        result.bytes_downloaded = row.bytes_downloaded
        results.append(result)

    stats = stage2(start=start, end=end)
    report = quality.build_report(
        results, stats, version_pairs=len(resolve_versions_across_quarters())
    )
    typer.echo(quality.render(report))


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
