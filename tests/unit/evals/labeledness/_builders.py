"""Screen and manifest builders shared by the harness tests.

A sibling module rather than conftest so the test modules can import it directly.
These directories carry no __init__.py, matching the rest of the suite, so a
relative import is not available and pytest's prepend import mode puts this
directory on sys.path instead.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime

from signaldesk.evals.labeledness.frame import FrameCounts
from signaldesk.evals.labeledness.manifest import (
    SampleManifest,
    Screen,
    SectionBlock,
    protocol_for,
)

ScreenFactory = Callable[..., Screen]


def build_screen(
    *,
    screen_id: str = "s0001",
    position: int = 1,
    pair_id: str = "DRUG||HEADACHE",
    drug_string: str = "DRUG",
    query: str = "DRUG",
    pt: str = "HEADACHE",
    set_id: str = "set-1",
    document_id: int = 1,
    blocks: Sequence[tuple[str, int, str]] | None = None,
    is_repeat: bool = False,
    repeat_of: str | None = None,
) -> Screen:
    items = list(blocks or [("adverse_reactions", 0, "Nausea and vomiting were reported.")])
    sections = tuple(
        SectionBlock.build(section_code=code, ordinal=ordinal, text=text)
        for code, ordinal, text in items
    )
    primary = sum(len(text) for code, _, text in items if code == "adverse_reactions")
    return Screen(
        screen_id=screen_id,
        position=position,
        pair_id=pair_id,
        drug_string=drug_string,
        query=query,
        pt=pt,
        set_id=set_id,
        document_id=document_id,
        protocol=protocol_for(primary),
        primary_chars=primary,
        sections=sections,
        is_repeat=is_repeat,
        repeat_of=repeat_of,
    )


def build_manifest(screens: Sequence[Screen], *, seed: int = 1234) -> SampleManifest:
    return SampleManifest(
        run_id="20260902T000000Z",
        written_at=datetime(2026, 9, 2, tzinfo=UTC),
        seed=seed,
        guideline_version="v1",
        signal_run_id="20260831T090758Z",
        signal_artifact="signals_20260831T091650Z.json",
        spl_artifact="spl_ingest_20260901T114909Z.json",
        signal_partitions_on_disk=1,
        frame=FrameCounts(
            documents_total=13205,
            documents_with_primary_section=9165,
            documents_without_primary_section=4040,
            strings_total=191,
            strings_clean=188,
            strings_capped=3,
            strings_unknown=0,
            strings_eligible=187,
            query_groups_eligible=157,
            documents_eligible=8878,
        ),
        frame_pairs=374846,
        screens=tuple(screens),
    )
