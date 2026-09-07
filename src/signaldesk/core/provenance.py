"""What a run needs to record about itself, wherever the run lives.

Two facts every committed artifact carries: the commit the code was at, and what
the run cost in memory. They were private to ``analytics.signals`` while it was
the only thing writing artifacts. The index build writes one too, and two copies
of a provenance helper is two things that can drift apart while both look
authoritative - so they live here and both callers import them.
"""

from __future__ import annotations

import os
import platform
import subprocess

from signaldesk.core.errors import SignalDeskError


def code_sha() -> str:
    """The commit the run happened at, or a marker saying it could not be read.

    ``.git`` is deliberately not mounted into the container - the application has
    no business reading the working tree - so ``git rev-parse`` cannot work there
    and the Makefile passes the host's answer in as ``SIGNALDESK_CODE_SHA``. The
    subprocess path is the fallback for a run invoked outside the container.

    An empty string here would look like a value. ``unknown`` says the provenance
    is incomplete, which is a different and worse thing than a run that has not
    been committed yet.
    """
    from_environment = os.environ.get("SIGNALDESK_CODE_SHA", "").strip()
    if from_environment:
        return from_environment

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def peak_rss_bytes() -> int:
    """Peak resident set size of this process.

    ``ru_maxrss`` is kilobytes on Linux and bytes on macOS. The container is
    Linux, and the multiplier is stated rather than guessed at read time.

    ``resource`` is a Unix-only module, so it is imported here rather than at
    module scope. Hassan develops on Windows and the unit suite runs there; an
    unconditional top-level import made the module holding this unimportable on
    that platform, which took every test that touches a build down with it. A
    build still cannot run on Windows - it raises below rather than reporting a
    peak of zero, because a fabricated measurement in a run record is worse than
    a refusal - but importing the module no longer depends on the platform.
    """
    try:
        import resource
    except ModuleNotFoundError as error:  # pragma: no cover - Windows only
        message = (
            "peak RSS cannot be measured on this platform: the 'resource' module "
            "is Unix-only. The pipeline runs in the Linux container; invoke the "
            "build through its make target rather than on the host."
        )
        raise SignalDeskError(message) from error

    usage = resource.getrusage(resource.RUSAGE_SELF)
    return int(usage.ru_maxrss) * (1 if platform.system() == "Darwin" else 1024)
