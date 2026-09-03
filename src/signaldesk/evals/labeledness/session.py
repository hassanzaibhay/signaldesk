"""The interactive loop: one screen, one keypress, one fsync.

Imports nothing from Django, DuckDB or any model client, and the test that pins
that walks this module's import graph statically rather than at runtime, so an
import buried inside a function is caught as well as one at the top of a file.
The remaining gap is a subprocess or an ``importlib`` call on a computed name;
neither is closed by that test and neither is claimed to be.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final
from typing import Protocol as TypingProtocol

from signaldesk.core.errors import AnnotationError
from signaldesk.core.logging import get_logger
from signaldesk.evals.labeledness.manifest import SampleManifest, Screen
from signaldesk.evals.labeledness.render import (
    ScreenView,
    render_note_prompt,
    render_screen,
    visible_sections,
)
from signaldesk.evals.labeledness.store import (
    AnnotationStore,
    Record,
    Verdict,
    next_position,
    undo_record,
    verdict_record,
)

log = get_logger(__name__)

#: Guideline section 8. Annotation stops here for the throughput and guideline
#: review, and only a deliberate flag carries past it.
CHECKPOINT_AT: Final = 50

VERDICT_KEYS: Final[dict[str, Verdict]] = {
    "l": Verdict.LABELLED_EXPLICIT,
    "b": Verdict.LABELLED_BROADER,
    "c": Verdict.LABELLED_CLASS,
    "n": Verdict.NOT_LABELLED,
    "u": Verdict.UNCLEAR,
}


class KeyReader(TypingProtocol):
    """Where keystrokes come from. A seam so the loop is testable without a tty."""

    def read_key(self) -> str: ...

    def read_line(self, prompt: str) -> str: ...


class TerminalKeyReader:
    """Single keypresses from a real terminal, raw on POSIX, direct on Windows.

    Falls back to line mode when stdin is not a tty, which is what happens under
    ``docker compose exec -T``. The fallback is announced rather than silent: an
    annotator who has to press Enter after every verdict should know why, since
    it roughly doubles the keystrokes over 330 screens.
    """

    def __init__(self) -> None:
        self.interactive = sys.stdin.isatty()

    def read_key(self) -> str:
        if not self.interactive:
            return (sys.stdin.readline() or "q").strip()[:1] or "\n"
        if sys.platform == "win32":  # pragma: no cover - host-only path
            import msvcrt

            return msvcrt.getwch()
        return self._read_key_posix()

    def _read_key_posix(self) -> str:  # pragma: no cover - requires a tty
        import termios
        import tty

        descriptor = sys.stdin.fileno()
        saved = termios.tcgetattr(descriptor)
        try:
            tty.setraw(descriptor)
            return sys.stdin.read(1)
        finally:
            termios.tcsetattr(descriptor, termios.TCSADRAIN, saved)

    def read_line(self, prompt: str) -> str:
        sys.stdout.write(prompt)
        sys.stdout.flush()
        return (sys.stdin.readline() or "").rstrip("\n")


@dataclass
class SessionResult:
    """What one sitting did. Reported, not inferred."""

    answered: int
    undone: int
    stopped_at: int
    reached_checkpoint: bool
    finished: bool


def run_session(
    manifest: SampleManifest,
    store: AnnotationStore,
    keys: KeyReader,
    *,
    write: Callable[[str], None],
    past_checkpoint: bool = False,
    clock: Callable[[], float] = time.monotonic,
    guideline_version: str | None = None,
) -> SessionResult:
    """Annotate from the resume point until the annotator quits or the file ends.

    ``guideline_version`` is what goes on every record written here, defaulting to
    the manifest's. The two are different facts and conflating them was a
    modelling error: the manifest records the version the sample was *drawn*
    under, and a record records the version the verdict was *made* under. They
    agree until the guideline is amended, and the guideline's own amendment
    procedure in section 8 requires that they be able to disagree -- without this
    parameter no amendment could ever be recorded, because the manifest is
    committed and must not be rewritten to carry a version it was not drawn
    under.

    No verdict is preselected and no key defaults to one: an unrecognised key,
    Enter included, re-renders the screen and writes nothing. That is asserted by
    a test on the store rather than on the return value, because a default
    introduced downstream of this dispatch would still leave a record behind.
    """
    version = guideline_version or manifest.guideline_version
    screen_ids = [screen.screen_id for screen in manifest.screens]
    live = store.resolved()
    position = next_position(len(manifest.screens), live, screen_ids)
    answered = 0
    undone = 0
    last_answered: str | None = None

    while position <= len(manifest.screens):
        if not past_checkpoint and len(live) >= CHECKPOINT_AT:
            log.info("labeledness.session.checkpoint", answered=len(live))
            return SessionResult(
                answered=answered,
                undone=undone,
                stopped_at=position,
                reached_checkpoint=True,
                finished=False,
            )

        screen = manifest.screen_at(position)
        view = ScreenView()
        started = clock()

        while True:
            write(render_screen(screen, progress=_progress(position, manifest), view=view))
            key = keys.read_key()

            if key in VERDICT_KEYS:
                verdict = VERDICT_KEYS[key]
                note = None
                if verdict is Verdict.UNCLEAR:
                    note = keys.read_line(render_note_prompt()).strip() or None
                elapsed = max(0, int((clock() - started) * 1000))
                record = verdict_record(
                    screen_id=screen.screen_id,
                    verdict=verdict,
                    guideline_version=version,
                    protocol=screen.protocol,
                    elapsed_ms=elapsed,
                    set_id=screen.set_id,
                    document_id=screen.document_id,
                    section_codes=screen.section_codes,
                    digests=screen.digests(),
                    note=note,
                )
                store.append(record)
                live[screen.screen_id] = record
                last_answered = screen.screen_id
                answered += 1
                position += 1
                break

            if key == "z":
                # Within this session only. Retracting a verdict from an earlier
                # sitting would be retracting one the annotator cannot see, which
                # is worse than not offering it; correcting an old screen means
                # re-annotating it, and the resume scan picks it up once it is
                # retracted.
                target = last_answered
                if target is None:
                    continue
                store.append(undo_record(screen_id=target, guideline_version=version))
                live.pop(target, None)
                last_answered = None
                undone += 1
                position = next_position(len(manifest.screens), live, screen_ids)
                break

            if key == "q":
                return SessionResult(
                    answered=answered,
                    undone=undone,
                    stopped_at=position,
                    reached_checkpoint=False,
                    finished=False,
                )

            view = _navigate(key, screen, view, keys)

    return SessionResult(
        answered=answered,
        undone=undone,
        stopped_at=position,
        reached_checkpoint=False,
        finished=True,
    )


def _progress(position: int, manifest: SampleManifest) -> str:
    """Position in the schedule and nothing else.

    Not a verdict tally, not a repeat marker, not an elapsed average. Every one
    of those correlates with something the annotator is not supposed to be
    reasoning about.
    """
    return f"screen {position} of {len(manifest.screens)}"


def _navigate(key: str, screen: Screen, view: ScreenView, keys: KeyReader) -> ScreenView:
    """Paging, section switching and search. Never writes a record."""
    sections = visible_sections(screen)
    if key in {"1", "2", "3", "4"}:
        wanted = ("adverse_reactions", "boxed_warning", "warnings_and_cautions", "warnings")[
            int(key) - 1
        ]
        if wanted in sections:
            return view.with_section(sections.index(wanted))
        return view
    if key == " ":
        return view.with_page(view.page + 1)
    if key == "B":
        return view.with_page(view.page - 1)
    if key == "G":
        return view.with_page(0)
    if key == "/":
        return view.with_query(keys.read_line("  find: ").strip())
    # Match stepping is on "." and ",". n and N are not available: n is a verdict
    # key, and overloading it onto navigation turns a mistimed keypress into a
    # written label.
    if key == "." and view.query:
        return replace(view, match_index=view.match_index + 1)
    if key == "," and view.query:
        return replace(view, match_index=max(0, view.match_index - 1))
    return view


def load_session(manifest_path: Path, gold_path: Path) -> tuple[SampleManifest, AnnotationStore]:
    """Open a manifest and its store, refusing a mismatched pair.

    The store is keyed on screen ids from one manifest. Pointing it at a second
    draw would resume against ids that mean something else, so a record naming a
    screen this manifest does not have stops the session.
    """
    from signaldesk.evals.labeledness.manifest import read_manifest

    manifest = read_manifest(manifest_path)
    store = AnnotationStore(gold_path)
    known = {screen.screen_id for screen in manifest.screens}
    stray = sorted({record.screen_id for record in store.read_all()} - known)
    if stray:
        message = (
            f"{gold_path} holds records for screens absent from {manifest_path.name}: "
            f"{', '.join(stray[:5])}. The store belongs to a different draw."
        )
        raise AnnotationError(message)
    return manifest, store


#: Fewest live verdicts the evaluation will score. Tied to the checkpoint rather
#: than chosen separately, so the two cannot drift apart.
#:
#: This closes the second gap left open when hygiene rule 3 was widened to
#: evals/golden/. That rule sees untracked files; it cannot see a gold set that
#: was never written, or one committed and then emptied, because both present as
#: a clean run. Scoring is where that gets caught instead: a missing or thin gold
#: set fails loudly rather than producing a metric over nothing. Below the
#: checkpoint the guideline gate has not been passed either, so any records there
#: were made under a guideline nobody has reviewed against measured behaviour.
MINIMUM_SCORABLE_RECORDS: Final = CHECKPOINT_AT


def require_scorable(
    manifest: SampleManifest,
    store: AnnotationStore,
    *,
    minimum: int = MINIMUM_SCORABLE_RECORDS,
) -> list[Record]:
    """Return the live verdicts, or refuse to let anything be scored.

    Three refusals, all loud: the file does not exist, it holds no live verdicts,
    or it holds fewer than ``minimum``. An evaluation that quietly reports zero
    against an absent gold set is worse than one that fails, because zero looks
    like a measurement.
    """
    if not store.path.exists():
        message = (
            f"no gold set at {store.path}. The evaluation will not score an absent "
            "gold set; annotate first, or point --gold at the right file."
        )
        raise AnnotationError(message)
    live = list(store.resolved().values())
    if not live:
        message = (
            f"{store.path} holds no live verdicts. Every record in it is retracted or "
            "the file is empty; there is nothing to score."
        )
        raise AnnotationError(message)
    if len(live) < minimum:
        message = (
            f"{store.path} holds {len(live)} live verdicts, fewer than the {minimum} "
            "required to score. That threshold is the screen-50 checkpoint: below it "
            "the guideline has not been reviewed against measured behaviour, so the "
            "records were made under a guideline nobody has checked."
        )
        raise AnnotationError(message)
    return verified(manifest, live)


def verified(manifest: SampleManifest, records: list[Record]) -> list[Record]:
    """Drop nothing, but refuse to return records whose digests have moved.

    The check the evaluation loader runs before scoring. A verdict whose recorded
    digests disagree with the manifest was made against text that is no longer
    what the manifest says, and scoring it would attribute a judgement to
    material nobody saw.
    """
    by_id = {screen.screen_id: screen for screen in manifest.screens}
    for record in records:
        if record.kind != "verdict":
            continue
        screen = by_id.get(record.screen_id)
        if screen is None:
            message = f"record names unknown screen {record.screen_id}"
            raise AnnotationError(message)
        if record.digests != screen.digests():
            message = (
                f"screen {record.screen_id} was annotated against different text than the "
                "manifest carries; refusing to score it"
            )
            raise AnnotationError(message)
    return records
