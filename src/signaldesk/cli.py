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
from typing import TYPE_CHECKING, Annotated, NoReturn

import typer

from signaldesk import __version__
from signaldesk.core.logging import configure_logging, get_logger

if TYPE_CHECKING:
    # Type-only: importing these for real pulls in the ORM, and the app registry
    # is not loaded until a command that needs it calls _setup_django.
    from signaldesk.ingest.faers.dedup import ChainReport
    from signaldesk.ingest.faers.pipeline import QuarterResult
    from signaldesk.ingest.faers.quarter import Quarter

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


def _results_from_manifest(
    start: Quarter | None = None, end: Quarter | None = None
) -> list[QuarterResult]:
    """Rebuild per-quarter results from the manifest.

    The manifest is what the ingest actually recorded, so a report built from it
    describes the corpus on disk rather than the arguments of the current call.
    """
    from signaldesk.ingest.faers.pipeline import QuarterResult
    from signaldesk.ingest.faers.quarter import Quarter
    from signaldesk.web.signals.models import IngestManifest

    results: list[QuarterResult] = []
    rows = IngestManifest.objects.filter(source="faers", status="completed").order_by("unit")
    for row in rows:
        quarter = Quarter.parse(row.unit)
        if (start is not None and quarter < start) or (end is not None and quarter > end):
            continue
        result = QuarterResult(quarter=quarter)
        result.row_counts = dict(row.row_counts or {})
        result.had_deleted_file = row.had_deleted_file
        result.bytes_downloaded = row.bytes_downloaded
        results.append(result)
    return results


def _postgres_cases() -> int:
    """Cases as the system of record counts them, one row per case.

    The duplicate figures are counted in Postgres, so their denominator is too.
    The analytical store holds one row per publication and is 707 rows larger;
    dividing one by the other is the store mixing ADR 0004 warns about.
    """
    from signaldesk.web.signals.models import Case

    return int(Case.objects.count())


def _chain_report() -> ChainReport:
    """Resolve canonical pointers before any rate is written."""
    from signaldesk.ingest.faers.dedup import chain_report

    return chain_report()


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

    report = quality.build_report(
        results,
        stats,
        version_pairs=len(version_pairs),
        postgres_cases=_postgres_cases() if stats is not None else None,
        chains=_chain_report() if stats is not None else None,
    )
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
    from signaldesk.analytics import faers as analytics
    from signaldesk.ingest.faers import quality
    from signaldesk.ingest.faers.dedup import persist, resolve_versions_across_quarters, stage2
    from signaldesk.ingest.faers.quarter import Quarter

    start = Quarter.parse(from_quarter) if from_quarter else None
    end = Quarter.parse(to_quarter) if to_quarter else None

    version_pairs = resolve_versions_across_quarters()
    stats = stage2(start=start, end=end)
    rows = persist(stats, version_pairs)

    # Write the report here rather than only from the ingest command: this pass
    # is where the deduplication numbers are measured, and an artifact assembled
    # anywhere else would be a transcription of them rather than a record.
    chains = _chain_report()
    report = quality.build_report(
        _results_from_manifest(start, end),
        stats,
        version_pairs=len(version_pairs),
        corpus=analytics.corpus_metrics(),
        postgres_cases=_postgres_cases(),
        chains=chains,
    )
    path = quality.write_report(report)

    typer.echo(f"population:                {stats.population:,}")
    typer.echo(f"of those, compared:        {stats.records_considered:,}")
    typer.echo(
        f"excluded, null block key:  {stats.records_excluded_null_key:,} ({stats.excluded_share:.2%})"
    )
    typer.echo(f"excluded, superseded:      {stats.records_excluded_superseded:,}")
    typer.echo(f"excluded, no drugs:        {stats.records_excluded_empty_drug_set:,}")
    typer.echo(f"blocks:                    {stats.blocks:,} (largest {stats.largest_block:,})")
    typer.echo(
        f"comparisons:               {stats.comparisons:,} (naive would be {stats.naive_comparisons:,})"
    )
    typer.echo(f"duplicate pairs:           {stats.duplicate_pairs:,}")
    typer.echo(f"cross-quarter share:       {stats.cross_quarter_share:.2%}")
    typer.echo(f"superseded across quarters:{len(version_pairs):,}")
    typer.echo(f"rows written:              {rows:,}")
    typer.echo(
        f"chains: {chains.closed_components:,} closed components, {chains.cycles:,} cycles"
        f" ({'sound' if chains.is_sound else 'NOT SOUND'})"
    )
    typer.echo(f"report written to {path}")


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
    recompute: Annotated[
        bool,
        typer.Option(
            "--recompute",
            help="Run deduplication again instead of using stored results. Takes about an hour.",
        ),
    ] = False,
) -> None:
    """Write and print the quality report for what is loaded.

    Uses the last recorded pass, or the stored corpus and duplicate table, so
    the report is a fast rebuild rather than an hour of recomputation. Pass
    --recompute to measure again; figures only a pass can produce are reported
    as unavailable otherwise.
    """
    _setup_django()
    from signaldesk.analytics import faers as analytics
    from signaldesk.ingest.faers import quality
    from signaldesk.ingest.faers.dedup import (
        load_stats,
        resolve_versions_across_quarters,
        save_stats,
        stage2,
        stats_from_store,
    )
    from signaldesk.ingest.faers.quarter import Quarter

    start = Quarter.parse(from_quarter) if from_quarter else None
    end = Quarter.parse(to_quarter) if to_quarter else None

    results = _results_from_manifest(start, end)
    # Reuse the last pass rather than repeating an hour of comparison; only
    # recompute when no pass has been recorded, and record it when we do.
    if recompute:
        stats = stage2(start=start, end=end)
        save_stats(stats)
    else:
        stats = load_stats() or stats_from_store()
    report = quality.build_report(
        results,
        stats,
        version_pairs=len(resolve_versions_across_quarters()),
        corpus=analytics.corpus_metrics(),
        postgres_cases=_postgres_cases(),
        chains=_chain_report(),
    )
    path = quality.write_report(report)
    typer.echo(quality.render(report))
    typer.echo(f"\nreport written to {path}")


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
def signals_build(
    drug_key: Annotated[
        str,
        typer.Option(
            "--drug-key",
            help=(
                "What counts as one drug: 'ingredient' resolves through the RxNorm "
                "normalization tables, 'raw-string' groups on the published drug name "
                "without resolving it. Recorded on the run; no number means anything "
                "without it."
            ),
        ),
    ] = "ingredient",
    roles: Annotated[
        str,
        typer.Option(
            "--roles",
            help=(
                "Reported drug roles in scope: 'ps-ss' for the primary analysis, "
                "'ps' for the primary-suspect sensitivity analysis."
            ),
        ),
    ] = "ps-ss",
    min_a: Annotated[
        int,
        typer.Option(
            "--min-a",
            help=(
                "Minimum co-reporting cases for a pair to enter the headline counts. "
                "Pairs below it are still computed and written, marked insufficient."
            ),
        ),
    ] = 3,
    from_quarter: Annotated[
        str | None, typer.Option("--from", help="First quarter, for example 2012Q4.")
    ] = None,
    to_quarter: Annotated[str | None, typer.Option("--to", help="Last quarter.")] = None,
) -> None:
    """Recompute contingency tables and the four estimators over the corpus.

    Writes one immutable run to /data/parquet/signal, plus the parameters, the
    commit and the data snapshot that produced it. Takes a few minutes on the
    full corpus; the wall clock and peak memory are measured and reported.
    """
    _setup_django()

    from signaldesk.analytics.contingency import ContingencySpec, DrugKey, RoleFilter
    from signaldesk.analytics.signals import build as build_signals

    try:
        spec = ContingencySpec(
            drug_key=DrugKey(drug_key.replace("-", "_")),
            roles=RoleFilter(roles.replace("-", "_")),
            min_a=min_a,
            start_quarter=from_quarter,
            end_quarter=to_quarter,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error

    record, path = build_signals(spec)

    typer.echo(f"run:                 {record.run_id}")
    typer.echo(f"drug key:            {record.params['drug_key']}")
    typer.echo(f"roles:               {record.params['role_codes']}")
    typer.echo(f"deduplicated cases:  {record.n_cases:,}")
    typer.echo(f"pairs written:       {record.pairs_observed:,}")
    typer.echo(f"pairs with a >= {min_a}:   {record.pairs_sufficient:,}")
    typer.echo("flagged, over pairs at or above the minimum cell count:")
    for name in ("ror", "prr", "bcpnn", "three_of_four"):
        headline = record.flag_counts[name]
        raw = record.flag_counts[f"{name}_including_insufficient"]
        typer.echo(f"  {name:16} {headline:>12,}   (with insufficient pairs: {raw:,})")
    if record.mgps_provisional:
        typer.echo(
            "MGPS:                provisional - the prior converged onto a bound "
            f"({', '.join(record.hyperparameters['on_boundary'])}). "  # type: ignore[arg-type]
            "EBGM columns are written but are not measurements."
        )
    else:
        typer.echo(f"MGPS flagged:        {record.flag_counts['mgps']:,}")
        typer.echo(f"all four:            {record.flag_counts['all_four']:,}")
    typer.echo(f"seconds:             {record.seconds:,.1f}")
    typer.echo(f"peak RSS:            {record.peak_rss_bytes / 1024**3:.2f} GiB")
    typer.echo(f"written to:          {path}")


@signals_app.command("artifact")
def signals_artifact(
    runs: Annotated[
        list[str],
        typer.Option(
            "--run",
            help=(
                "Run identifier to include. Repeat for each run. The primary analysis "
                "and its sensitivity run belong in one artifact."
            ),
        ),
    ],
    diagnostics: Annotated[
        Path | None,
        typer.Option(
            "--mgps-diagnostic",
            help=(
                "JSON file holding the MGPS boundary diagnostic - the profile "
                "likelihood and the EBGM05 sensitivity across it - to embed in the "
                "artifact."
            ),
        ),
    ] = None,
) -> None:
    """Collect signal runs into one committed artifact under evals/history/."""
    _setup_django()

    from signaldesk.analytics.signals import write_history_artifact

    path = write_history_artifact(runs, diagnostics=diagnostics)
    typer.echo(f"written to {path}")


@index_app.command("build")
def index_build() -> None:
    """Chunk documents, embed Tier A, and build the sparse and dense indexes."""
    _owned_by("P10", "index build")


def _run_signals_suite() -> None:
    """Validate the estimators against the published reference standards.

    Refuses, naming every path, until the curated files exist. They are written
    by hand and are never generated here: a reference standard produced by the
    system it evaluates measures nothing.
    """
    from signaldesk.evals.signals.reference import ReferenceSetError, load_all

    try:
        _outcome_map, sets = load_all()
    except ReferenceSetError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error

    for name, reference in sets.items():
        typer.echo(
            f"{name}: {len(reference)} pairs ({reference.positives}+/{reference.negatives}-)"
        )


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
    if suite == "signals":
        _setup_django()
        _run_signals_suite()
        return
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
