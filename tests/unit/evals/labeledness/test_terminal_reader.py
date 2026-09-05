"""The real key reader, against a real terminal.

This file exists because it did not. Every other session test injects a scripted
key reader, so the actual input path -- isatty branching, buffering, raw mode,
end of input -- was never executed by the suite, and two 50-screen annotation
passes were discarded to a defect in it. The pace flag caught the consequence
both times; nothing caught the cause.
"""

from __future__ import annotations

import io
import os
import sys
import textwrap
from pathlib import Path
from typing import ClassVar

import pytest

from signaldesk.core.errors import EndOfInputError
from signaldesk.evals.labeledness.session import TerminalKeyReader

pytestmark = pytest.mark.unit

posix_only = pytest.mark.skipif(
    sys.platform == "win32", reason="pty and termios are POSIX; CI and the container are Linux"
)

#: Run inside a forked pty so ``isatty()`` is true and the POSIX branch executes.
#: Reading through a pipe would take the line-mode branch and prove nothing about
#: the one that failed.
CHILD = textwrap.dedent(
    """
    import sys, time
    sys.path.insert(0, {src!r})
    from signaldesk.evals.labeledness.session import TerminalKeyReader

    def mark(tag):
        sys.stderr.write("<%s>" % tag)
        sys.stderr.flush()

    reader = TerminalKeyReader()
    keys = []
    for index in range({reads}):
        if index:
            mark("GAP%d" % index)
            time.sleep({gap})          # stands in for rendering the next screen
        mark("READ%d" % index)
        keys.append(reader.read_key())
    sys.stderr.write("INTERACTIVE=%r KEYS=%r\\n" % (reader.interactive, keys))
    sys.stderr.flush()
    """
)

#: Margin between the child announcing a read and actually blocking in it. A few
#: Python statements; generous at this size and, unlike a startup delay, not
#: sensitive to how slowly the interpreter came up.
SETTLE = 0.4


def _drive(
    reads: int,
    steps: list[tuple[str, bytes]],
    timeout: float = 40.0,
    gap: float = 0.0,
) -> str:
    """Fork a pty, run the reader in the child, and write to it on cue.

    Each step is (marker, payload): wait until the child has printed that marker,
    settle briefly, then write. Waiting on the child's own markers rather than on
    elapsed time is what makes this survive a slow interpreter -- an earlier
    version used fixed delays and passed under `pytest -n auto` while failing
    under coverage, because the child had not reached its first read when the
    parent wrote.
    """
    import pty
    import select
    import time

    source = str(Path("src").resolve())
    code = CHILD.format(src=source, reads=reads, gap=gap)

    pid, master = pty.fork()
    if pid == 0:  # pragma: no cover - child process
        os.execv(sys.executable, [sys.executable, "-c", code])
        os._exit(1)

    captured = b""
    pending = list(steps)
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            if pending:
                marker, payload = pending[0]
                if f"<{marker}>".encode() in captured:
                    time.sleep(SETTLE)
                    os.write(master, payload)
                    pending.pop(0)
            ready, _, _ = select.select([master], [], [], 0.05)
            if ready:
                try:
                    chunk = os.read(master, 4096)
                except OSError:
                    break
                if not chunk:
                    break
                captured += chunk
                if b"KEYS=" in captured:
                    break
    finally:
        with contextlib_suppress():
            os.close(master)
        with contextlib_suppress():
            os.waitpid(pid, os.WNOHANG)
    return captured.decode(errors="replace")


class contextlib_suppress:  # noqa: N801 - a two-line helper, not a public class
    """Swallow teardown errors so a pty already closed cannot fail a test."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> bool:
        return True


@posix_only
class TestABurstDoesNotBecomeManyVerdicts:
    """The defect, stated as the property it violated.

    The old reader called ``sys.stdin.read(1)``. Its first call pulled a whole
    chunk off the descriptor into the ``TextIOWrapper`` buffer and every later
    call was served from that chunk, so six characters typed in one burst became
    six answered screens in a few milliseconds.
    """

    def test_six_characters_in_one_burst_yield_one_key(self) -> None:
        """One burst, two reads, and the second must wait for a fresh keystroke.

        The second read returns "b" rather than "n": the five leftover "n"s were
        discarded by the flush, not queued. Under the old reader both reads
        returned "n" and no wait occurred.

        WHAT WOULD HAVE TO REGRESS FOR THIS TO STAY GREEN WHILE THE DEFECT
        RETURNS: the assertion is on the VALUE of the second key, not on timing,
        so it survives a slow machine and cannot be satisfied by a buffered read.
        The one way to fool it is to write the second burst before the first read
        completes, which the delays in the script prevent -- the "b" is sent a
        second after the "n"s, by which time the first read has returned. If that
        ordering were removed, both bursts would merge into one queue and the
        flush would discard "b" too, turning this red rather than green.
        """
        out = _drive(reads=2, steps=[("READ0", b"nnnnnn"), ("READ1", b"b")])

        assert "INTERACTIVE=True" in out, out
        assert "KEYS=['n', 'b']" in out, out
        assert "KEYS=['n', 'n']" not in out

    def test_a_key_typed_while_the_screen_renders_is_discarded(self) -> None:
        """A verdict must come from a key pressed while the screen was up.

        The child sleeps between reads, standing in for the render, and the "xxxx"
        lands inside that sleep -- queued by the line discipline exactly as it is
        when a real screen is being drawn. The next read must discard it and wait
        for "u".

        WHAT THIS TEST DOES AND DOES NOT CATCH, stated because it is easy to
        overclaim: it would NOT have caught the shipped defect. ``tty.setraw``
        defaults to ``when=TCSAFLUSH``, which already discards pending input, so
        the old reader passed this too -- its buffer sat downstream of the flush
        and was never what the flush could reach. The explicit ``tcflush`` makes
        the discard a stated property rather than a side effect of a default
        argument, and this test is what holds if that default changes or if
        ``setraw`` is replaced with a hand-rolled ``tcsetattr``. The burst test
        above is the one that catches the defect that cost two sessions.
        """
        out = _drive(
            reads=2,
            gap=2.0,
            steps=[("READ0", b"a"), ("GAP1", b"xxxx"), ("READ1", b"u")],
        )

        assert "KEYS=['a', 'u']" in out, out
        assert "KEYS=['a', 'x']" not in out


@posix_only
class TestEndOfInputRaises:
    def test_the_line_mode_reader_raises_rather_than_returning_a_key(self) -> None:
        """The old reader answered EOF with "q", which quietly quit the session.

        WHAT WOULD HAVE TO REGRESS FOR THIS TO STAY GREEN WHILE THE DEFECT
        RETURNS: nothing subtle. The assertion is that a specific typed exception
        escapes, so any substituted keystroke -- "q" or otherwise -- fails it. It
        would only pass wrongly if EndOfInputError were raised somewhere else in
        the call and the read still returned a default, which cannot happen since
        the raise is the last statement on that path.
        """
        reader = TerminalKeyReader.__new__(TerminalKeyReader)
        reader.interactive = False
        original = sys.stdin
        sys.stdin = io.StringIO("")
        try:
            with pytest.raises(EndOfInputError, match="end of input"):
                reader.read_key()
        finally:
            sys.stdin = original

    def test_line_mode_still_reads_a_key_when_there_is_one(self) -> None:
        """The raise must not have made the ordinary path unreachable."""
        reader = TerminalKeyReader.__new__(TerminalKeyReader)
        reader.interactive = False
        original = sys.stdin
        sys.stdin = io.StringIO("n\nb\n")
        try:
            assert reader.read_key() == "n"
            assert reader.read_key() == "b"
            with pytest.raises(EndOfInputError):
                reader.read_key()
        finally:
            sys.stdin = original


class TestTheAnnotateCommandForwardsEveryOptionItParses:
    """The wiring, not the unit.

    ``--guideline-version`` was parsed into a local and never passed to
    ``run_session``, so every record carried the manifest's version while a test
    asserting the override passed cleanly -- because it called ``run_session``
    directly and the severed path was above it. Ruff does not help: an unused
    ARGUMENT is ARG001, which is not in the select list, and F841 covers locals
    only.

    So this asserts the path. It derives the option list from the command's own
    signature rather than naming them, which is what makes it hold for options
    added later.
    """

    #: Consumed by load_session before run_session is reached, so they are the
    #: only parameters legitimately absent from the forwarded call.
    CONSUMED_BEFORE_THE_SESSION: ClassVar[frozenset[str]] = frozenset(
        {"manifest_path", "gold_path"}
    )

    def _invoke(self, tmp_path: Path, extra: list[str]) -> dict[str, object]:
        import inspect

        from _builders import build_manifest, build_screen
        from typer.testing import CliRunner

        from signaldesk.cli import app
        from signaldesk.evals.labeledness import session as session_module
        from signaldesk.evals.labeledness.manifest import write_manifest

        manifest_path = tmp_path / "sample.json"
        write_manifest(build_manifest([build_screen()]), manifest_path)

        captured: dict[str, object] = {}

        def _fake_run_session(*args: object, **kwargs: object) -> session_module.SessionResult:
            captured.update(kwargs)
            return session_module.SessionResult(
                answered=0, undone=0, stopped_at=1, reached_checkpoint=False, finished=False
            )

        original = session_module.run_session
        session_module.run_session = _fake_run_session  # type: ignore[assignment]
        try:
            result = CliRunner().invoke(
                app,
                [
                    "evals",
                    "annotate",
                    "--manifest",
                    str(manifest_path),
                    "--gold",
                    str(tmp_path / "gold.jsonl"),
                    *extra,
                ],
            )
        finally:
            session_module.run_session = original  # type: ignore[assignment]

        assert result.exit_code == 0, result.output
        captured["__signature__"] = inspect.signature(_annotate_command()).parameters.keys()
        return captured

    def test_guideline_version_reaches_run_session(self, tmp_path: Path) -> None:
        """The specific defect: v2 asked for, v1 recorded."""
        captured = self._invoke(tmp_path, ["--guideline-version", "v2"])
        assert captured["guideline_version"] == "v2"

    def test_past_checkpoint_reaches_run_session(self, tmp_path: Path) -> None:
        captured = self._invoke(tmp_path, ["--past-checkpoint"])
        assert captured["past_checkpoint"] is True

    def test_every_parsed_option_is_forwarded(self, tmp_path: Path) -> None:
        """The general form, so a future option cannot be dropped the same way.

        WHAT WOULD HAVE TO REGRESS FOR THIS TO STAY GREEN WHILE AN OPTION IS
        DROPPED: the option would have to be added to
        CONSUMED_BEFORE_THE_SESSION, which is a deliberate edit naming it, or be
        consumed by load_session without being listed there -- in which case this
        goes red and the fix is to list it, with the listing itself recording
        that the option never reaches the session. It cannot be satisfied by
        forwarding a differently named key, because the names come from the
        command's own signature.
        """
        captured = self._invoke(tmp_path, ["--guideline-version", "v2", "--past-checkpoint"])
        parsed = set(captured.pop("__signature__"))  # type: ignore[arg-type]
        forwarded = set(captured)

        missing = parsed - self.CONSUMED_BEFORE_THE_SESSION - forwarded
        assert missing == set(), f"parsed but never forwarded to run_session: {sorted(missing)}"


def _annotate_command() -> object:
    """The undecorated annotate callback, for signature introspection."""
    from signaldesk.cli import evals_annotate

    return evals_annotate
