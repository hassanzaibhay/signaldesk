"""The one-shot seeded draw.

Separate module from the annotation path on purpose. This one reads Postgres and
the signal parquet; nothing under :mod:`signaldesk.evals.labeledness.session`
imports it, which is what lets the harness run with no database.

The draw runs once. Its seed and every count it saw go into the committed
manifest, so a reviewer can re-run it and get the same 330 screens, and so the
frame cannot be adjusted after seeing what came out.
"""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from signaldesk.core.errors import AnnotationError
from signaldesk.core.logging import get_logger
from signaldesk.evals.labeledness.frame import (
    PRIMARY_SECTION,
    SECTION_ORDER,
    DocumentRow,
    Frame,
    ReachRow,
    build_frame,
    read_fetch_units,
)
from signaldesk.evals.labeledness.manifest import (
    SampleManifest,
    Screen,
    SectionBlock,
    manifest_filename,
    new_run_id,
    protocol_for,
)
from signaldesk.evals.labeledness.render import content_words, stem
from signaldesk.evals.labeledness.schedule import (
    TOTAL_UNIQUE,
    build_schedule,
    first_presentation_index,
)

log = get_logger(__name__)

MIN_COUNT: Final = 3
RESERVE_TARGET: Final = 100
#: How many candidates the reserve scan may look at before giving up. Bounded so
#: the draw terminates on a corpus where the lexical predicate is rarer than
#: measured, rather than walking 374,846 pairs.
RESERVE_SCAN_LIMIT: Final = 20_000


@dataclass(frozen=True, slots=True)
class Pair:
    """One deduplicated (query group, Preferred Term) pair."""

    query: str
    pt: str

    @property
    def pair_id(self) -> str:
        return f"{self.query}||{self.pt}"


def load_documents_and_reach() -> tuple[list[DocumentRow], list[ReachRow]]:
    """Read the label corpus down to what the frame decides on.

    Deliberately two flat lists rather than an ORM traversal: the frame is pure
    and testable, and everything database-shaped stops here.
    """
    from django.db.models import Count, Q

    from signaldesk.web.documents.models import LabelDocument, LabelDrugKey

    documents = [
        DocumentRow(
            document_id=row["id"],
            set_id=row["set_id"],
            has_primary_section=row["primary_blocks"] > 0,
        )
        for row in LabelDocument.objects.annotate(
            primary_blocks=Count("sections", filter=Q(sections__section_code=PRIMARY_SECTION))
        ).values("id", "set_id", "primary_blocks")
    ]
    reach = [
        ReachRow(
            folded_string=row["folded_string"], query=row["query"], document_id=row["document"]
        )
        for row in LabelDrugKey.objects.values("folded_string", "query", "document")
    ]
    return documents, reach


def signal_partitions(settings: object | None = None) -> list[str]:
    """Every signal run partition on disk, ascending.

    Read and recorded rather than assumed. The SPL ingest artifact does not say
    which signal run its scope came from, so the link between the drug strings
    that were fetched and the run that flagged them rests on there having been
    exactly one partition. That is now a checked fact carried in the manifest
    instead of an unstated premise.
    """
    from signaldesk.analytics.signals import signal_root

    root = signal_root(settings)  # type: ignore[arg-type]
    if not root.exists():
        return []
    return sorted(path.name.removeprefix("run=") for path in root.iterdir() if path.is_dir())


def load_pairs(run_id: str, frame: Frame, settings: object | None = None) -> list[Pair]:
    """Deduplicated (query group, PT) pairs at or above the minimum count.

    Deduplication is the third frame ruling: ``PREDNISONE`` and ``PREDNISONE.``
    clean to one openFDA query and reach an identical document set, so for a
    labeledness verdict they are one question asked twice. Left in, they double
    the sampling mass of the 29 groups that have a sibling.
    """
    import duckdb

    from signaldesk.analytics.signals import signal_root

    partition = signal_root(settings) / f"run={run_id}"  # type: ignore[arg-type]
    files = sorted(partition.glob("*.parquet"))
    if not files:
        message = f"no parquet under {partition}"
        raise AnnotationError(message)

    connection = duckdb.connect()
    try:
        connection.execute("CREATE TABLE eligible(drug VARCHAR, grp VARCHAR)")
        connection.executemany(
            "INSERT INTO eligible VALUES (?, ?)", list(frame.query_by_string.items())
        )
        sources = ", ".join(f"'{path.as_posix()}'" for path in files)
        rows = connection.execute(
            f"SELECT DISTINCT grp, pt FROM read_parquet([{sources}]) "
            "JOIN eligible USING(drug) WHERE a >= ? AND pt IS NOT NULL AND trim(pt) <> '' "
            "ORDER BY grp, pt",
            [MIN_COUNT],
        ).fetchall()
    finally:
        connection.close()
    return [Pair(query=str(group), pt=str(pt)) for group, pt in rows]


def load_sections(document_ids: Sequence[int]) -> dict[int, list[tuple[str, int, str]]]:
    """Every block of the four read sections, for the drawn documents only."""
    from signaldesk.web.documents.models import LabelSection

    blocks: dict[int, list[tuple[str, int, str]]] = {}
    rows = LabelSection.objects.filter(
        document_id__in=list(document_ids), section_code__in=list(SECTION_ORDER)
    ).values("document_id", "section_code", "ordinal", "text")
    for row in rows:
        blocks.setdefault(row["document_id"], []).append(
            (row["section_code"], row["ordinal"], row["text"])
        )
    order = {code: index for index, code in enumerate(SECTION_ORDER)}
    for document_id, items in blocks.items():
        blocks[document_id] = sorted(items, key=lambda item: (order[item[0]], item[1]))
    return blocks


def has_lexical_evidence(pt: str, primary_text: str) -> bool:
    """The reserve stratum's predicate: explicit lexical evidence, nothing more.

    Exact Preferred Term, or every content word present stemmed. Published here
    because it decides membership of a committed stratum, and a stratum whose
    membership rule lives only in a run is not reproducible.

    It is not a labelledness judgement and is never shown to the annotator. It
    catches the explicit route only: subsumption and class warnings produce no
    lexical hit at all, which is exactly why the reserve is a supplement to a
    uniform sample and not a replacement for one.
    """
    haystack = primary_text.lower()
    if pt.lower() in haystack:
        return True
    words = content_words(pt)
    return bool(words) and all(stem(word) in haystack for word in words)


def _build_screen(
    *,
    screen_id: str,
    position: int,
    pair: Pair,
    drug_string: str,
    document_id: int,
    set_id: str,
    blocks: Sequence[tuple[str, int, str]],
    is_repeat: bool = False,
    repeat_of: str | None = None,
) -> Screen:
    sections = tuple(
        SectionBlock.build(section_code=code, ordinal=ordinal, text=text)
        for code, ordinal, text in blocks
    )
    primary_chars = sum(len(text) for code, _, text in blocks if code == PRIMARY_SECTION)
    return Screen(
        screen_id=screen_id,
        position=position,
        pair_id=pair.pair_id,
        drug_string=drug_string,
        query=pair.query,
        pt=pair.pt,
        set_id=set_id,
        document_id=document_id,
        protocol=protocol_for(primary_chars),
        primary_chars=primary_chars,
        sections=sections,
        is_repeat=is_repeat,
        repeat_of=repeat_of,
    )


def _primary_string(frame: Frame, query: str) -> str:
    """The member string a pair is presented under: first ascending.

    Matches the tie-break the scope selection already uses, so the choice is one
    convention across the project rather than two.
    """
    members = frame.strings_by_query[query]
    return members[0]


def _sibling_string(frame: Frame, query: str) -> str:
    """The string a repeat is presented under, when the group has a second one.

    Same query, same document, same event, different surface. Covers the 29
    groups of 157 that carry a trailing-period twin; where there is no sibling
    the repeat is presented identically, and the coverage is measured rather than
    claimed.
    """
    members = frame.strings_by_query[query]
    return members[1] if len(members) > 1 else members[0]


@dataclass(frozen=True, slots=True)
class DrawResult:
    manifest: SampleManifest
    path: Path


def draw(
    *,
    seed: int,
    guideline_version: str,
    spl_artifact: Path,
    signal_artifact: str,
    history_dir: Path,
    settings: object | None = None,
    run_id: str | None = None,
) -> DrawResult:
    """Build the frame, draw 300 plus the reserve, and write the manifest."""
    partitions = signal_partitions(settings)
    if not partitions:
        message = "no signal run partition on disk; build signals before drawing"
        raise AnnotationError(message)
    signal_run_id = partitions[-1]

    documents, reach = load_documents_and_reach()
    units = read_fetch_units(spl_artifact)
    frame = build_frame(documents=documents, reach=reach, units=units)
    pairs = load_pairs(signal_run_id, frame, settings)
    if len(pairs) < TOTAL_UNIQUE:
        message = f"frame holds {len(pairs)} pairs, fewer than the {TOTAL_UNIQUE} to be drawn"
        raise AnnotationError(message)

    rng = random.Random(seed)
    primary = rng.sample(pairs, TOTAL_UNIQUE)
    document_for = {
        pair.pair_id: rng.choice(frame.documents_for_query(pair.query)) for pair in primary
    }

    reserve_pairs, reserve_documents = _draw_reserve(
        pairs=pairs, taken={pair.pair_id for pair in primary}, frame=frame, rng=rng
    )

    wanted = sorted(set(document_for.values()) | set(reserve_documents.values()))
    blocks = load_sections(wanted)
    placements = build_schedule(TOTAL_UNIQUE, rng)
    firsts = first_presentation_index(placements)

    screens: list[Screen] = []
    screen_id_at_position: dict[int, str] = {}
    for placement in placements:
        pair = primary[placement.pair_index]
        document_id = document_for[pair.pair_id]
        screen_id = f"s{placement.position:04d}"
        screen_id_at_position[placement.position] = screen_id
        screens.append(
            _build_screen(
                screen_id=screen_id,
                position=placement.position,
                pair=pair,
                drug_string=(
                    _sibling_string(frame, pair.query)
                    if placement.is_repeat
                    else _primary_string(frame, pair.query)
                ),
                document_id=document_id,
                set_id=frame.set_id_by_document[document_id],
                blocks=blocks.get(document_id, []),
                is_repeat=placement.is_repeat,
                repeat_of=(
                    screen_id_at_position[firsts[placement.pair_index]]
                    if placement.is_repeat
                    else None
                ),
            )
        )

    reserve = tuple(
        _build_screen(
            screen_id=f"r{index + 1:04d}",
            position=index + 1,
            pair=pair,
            drug_string=_primary_string(frame, pair.query),
            document_id=reserve_documents[pair.pair_id],
            set_id=frame.set_id_by_document[reserve_documents[pair.pair_id]],
            blocks=blocks.get(reserve_documents[pair.pair_id], []),
        )
        for index, pair in enumerate(reserve_pairs)
    )

    from datetime import UTC, datetime

    from signaldesk.evals.labeledness.manifest import ReserveNote, write_manifest

    identifier = run_id or new_run_id()
    manifest = SampleManifest(
        run_id=identifier,
        written_at=datetime.now(tz=UTC),
        seed=seed,
        guideline_version=guideline_version,
        signal_run_id=signal_run_id,
        signal_artifact=signal_artifact,
        spl_artifact=spl_artifact.name,
        signal_partitions_on_disk=len(partitions),
        frame=frame.counts,
        frame_pairs=len(pairs),
        screens=tuple(screens),
        reserve=reserve,
        reserve_note=ReserveNote().text,
        notes=(
            "Sibling-string presentation covers "
            f"{sum(1 for query in frame.strings_by_query.values() if len(query) > 1)} of "
            f"{len(frame.strings_by_query)} query groups.",
            "page_cap null is excluded as unknown, never read as not-capped. On this "
            "corpus no live unit carries null, so the rule is non-binding here and "
            "binds on the next unforced ingest, where the attach path can produce one.",
        ),
    )
    path = history_dir / manifest_filename(identifier)
    written = write_manifest(manifest, path)
    log.info(
        "labeledness.draw.written",
        path=str(path),
        bytes=written,
        screens=len(screens),
        reserve=len(reserve),
        seed=seed,
    )
    return DrawResult(manifest=manifest, path=path)


def _draw_reserve(
    *, pairs: Sequence[Pair], taken: set[str], frame: Frame, rng: random.Random
) -> tuple[list[Pair], dict[str, int]]:
    """Scan a shuffled candidate pool for pairs carrying explicit lexical evidence."""
    candidates = [pair for pair in pairs if pair.pair_id not in taken]
    rng.shuffle(candidates)
    scanned = candidates[:RESERVE_SCAN_LIMIT]
    assigned = {pair.pair_id: rng.choice(frame.documents_for_query(pair.query)) for pair in scanned}
    texts = _primary_texts(sorted(set(assigned.values())))

    chosen: list[Pair] = []
    documents: dict[str, int] = {}
    for pair in scanned:
        if len(chosen) >= RESERVE_TARGET:
            break
        document_id = assigned[pair.pair_id]
        if has_lexical_evidence(pair.pt, texts.get(document_id, "")):
            chosen.append(pair)
            documents[pair.pair_id] = document_id
    return chosen, documents


def _primary_texts(document_ids: Sequence[int]) -> dict[int, str]:
    from signaldesk.web.documents.models import LabelSection

    joined: dict[int, list[str]] = {}
    rows = (
        LabelSection.objects.filter(
            document_id__in=list(document_ids), section_code=PRIMARY_SECTION
        )
        .order_by("document_id", "ordinal")
        .values("document_id", "text")
    )
    for row in rows:
        joined.setdefault(row["document_id"], []).append(row["text"])
    return {key: " ".join(value) for key, value in joined.items()}


def normalise_seed(raw: str) -> int:
    """Accept a decimal seed and reject anything a shell might have mangled."""
    if not re.fullmatch(r"\d{1,19}", raw):
        message = f"seed must be a decimal integer, got {raw!r}"
        raise AnnotationError(message)
    return int(raw)
