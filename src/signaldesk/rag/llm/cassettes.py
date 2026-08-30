"""Recorded provider responses, so the suite runs with no keys.

Continuous integration has no API keys at all. Every test in this layer
therefore replays from disk, and a test that needs a live key is a defect rather
than an inconvenience.

Replay is served through an ``httpx`` transport rather than by stubbing the
provider. That matters: a higher-level stub would return a ``RawCompletion`` and
skip the request building and response parsing, which is most of what a provider
module is. Going through the transport means the provider's own code runs
against the recorded bytes.

## These cassettes are constructed, and that is the weak point

Every cassette committed with this prompt was written from vendor documentation,
not captured from a provider. Nothing here has spoken to Gemini, Groq, Cerebras
or Ollama. If a response shape is wrong, the suite is green and the router is
fiction - and usage blocks and safety blocks are exactly where these APIs drift.

So every cassette records whether it was constructed or captured, and
``record-cassettes`` writes a marker when a real recording run has happened.
Until that marker exists, the test that would fail on a still-constructed
cassette skips instead of passing, because a green tick that means "nobody has
checked" is worse than a skip that says so.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import httpx

from signaldesk.core.logging import get_logger
from signaldesk.rag.llm.base import ChatRequest
from signaldesk.rag.llm.errors import CassetteMissError

log = get_logger(__name__)

#: Repository root, from this module: llm -> rag -> signaldesk -> src.
REPO_ROOT = Path(__file__).resolve().parents[4]

MODE_ENV_VAR = "SIGNALDESK_CASSETTE_MODE"

#: Written by a recording run. Its presence is what promotes the
#: still-constructed check from skipped to enforced.
MARKER_NAME = "recorded.json"


class Mode(StrEnum):
    """How a call reaches a provider."""

    REPLAY = "replay"
    RECORD = "record"
    OFF = "off"


def cassette_root() -> Path:
    """Where cassettes live. Repository data, not configurable."""
    return REPO_ROOT / "evals" / "cassettes"


def marker_path(root: Path | None = None) -> Path:
    return (root or cassette_root()) / MARKER_NAME


def current_mode() -> Mode:
    """The mode for this process. Replay unless told otherwise.

    Defaulting to replay is what makes an accidentally live test impossible: a
    developer who has keys in their environment still gets disk.
    """
    raw = os.environ.get(MODE_ENV_VAR, Mode.REPLAY).strip().lower()
    try:
        return Mode(raw)
    except ValueError:
        log.warning("llm.cassette.unknown_mode", requested=raw, using=str(Mode.REPLAY))
        return Mode.REPLAY


def key_for(provider: str, model: str, prompt_version: str, chat: ChatRequest) -> str:
    """A stable key over everything that would change the answer.

    The rendered messages are included, not the template name, because two calls
    to one prompt version with different inputs are different interactions. The
    schema name is included because the same messages under a different schema
    ask the provider for a different shape.
    """
    canonical = json.dumps(
        {
            "provider": provider,
            "model": model,
            "prompt_version": prompt_version,
            "schema": chat.schema_name,
            "messages": [
                {"role": message.role, "content": message.content} for message in chat.messages
            ],
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True, slots=True)
class Cassette:
    """One recorded interaction."""

    key: str
    provider: str
    model: str
    prompt_version: str
    schema_name: str
    status_code: int
    body: dict[str, Any]
    #: True when written by hand from documentation rather than captured from a
    #: provider. The whole point of the field is that it is loud.
    constructed: bool
    constructed_at: str = ""
    recorded_at: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Cassette:
        return cls(
            key=str(payload["key"]),
            provider=str(payload["provider"]),
            model=str(payload["model"]),
            prompt_version=str(payload.get("prompt_version", "")),
            schema_name=str(payload.get("schema_name", "")),
            status_code=int(payload.get("status_code", 200)),
            body=dict(payload.get("body") or {}),
            constructed=bool(payload.get("constructed", True)),
            constructed_at=str(payload.get("constructed_at", "")),
            recorded_at=payload.get("recorded_at"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "provider": self.provider,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "schema_name": self.schema_name,
            "status_code": self.status_code,
            "constructed": self.constructed,
            "constructed_at": self.constructed_at,
            "recorded_at": self.recorded_at,
            "body": self.body,
        }


def path_for(key: str, root: Path | None = None) -> Path:
    return (root or cassette_root()) / f"{key}.json"


def load(key: str, root: Path | None = None) -> Cassette:
    """Read one cassette, or raise. Replay never falls back to the network."""
    path = path_for(key, root)
    if not path.is_file():
        raise CassetteMissError(key, path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CassetteMissError(key, path, detail=f"unreadable: {error}") from error
    return Cassette.from_dict(payload)


def save(cassette: Cassette, root: Path | None = None) -> Path:
    """Write one cassette, human-readable and stable under diff."""
    path = path_for(cassette.key, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline is pinned, not left to the platform. Python's text mode
    # translates to the local ending, so a recording run on Windows would write
    # CRLF into files that are committed, and the portability gate rejects CRLF
    # in tracked text. The cassette is repository content, not local output.
    path.write_text(
        json.dumps(cassette.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def all_cassettes(root: Path | None = None) -> list[Cassette]:
    """Every cassette on disk, marker and placeholder excluded."""
    directory = root or cassette_root()
    if not directory.is_dir():
        return []
    found = []
    for path in sorted(directory.glob("*.json")):
        if path.name == MARKER_NAME:
            continue
        try:
            found.append(Cassette.from_dict(json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, json.JSONDecodeError, KeyError):
            log.warning("llm.cassette.unreadable", path=str(path))
    return found


def replay_transport(cassette: Cassette) -> httpx.MockTransport:
    """A transport that answers any request with this cassette's recorded body.

    Scoped to one interaction: the router resolves the key before the call and
    builds a transport for exactly that cassette, so a wrong key is a miss
    rather than a silently reused response from another interaction.
    """

    def _handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=cassette.status_code, json=cassette.body)

    return httpx.MockTransport(_handler)


def write_marker(
    recorded: list[str], differences: dict[str, list[str]], root: Path | None = None
) -> Path:
    """Record that a real recording run happened, and what it changed.

    ``differences`` maps a cassette key to the fields where the captured body
    differed from the constructed one. That list is the point of the exercise:
    it is the evidence about which parts of the hand-written shapes were wrong.
    """
    path = marker_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "recorded_at": datetime.now(tz=UTC).isoformat(),
        "keys": sorted(recorded),
        "fields_that_differed": {key: sorted(value) for key, value in sorted(differences.items())},
        "note": (
            "Written by 'signaldesk evals record-cassettes'. Its presence means at "
            "least one cassette has been captured from a live provider, which "
            "promotes the still-constructed check from skipped to enforced."
        ),
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    log.info("llm.cassette.marker_written", path=str(path), keys=len(recorded))
    return path


def has_been_recorded(root: Path | None = None) -> bool:
    """Whether any recording run has ever happened."""
    return marker_path(root).is_file()


def differing_fields(constructed: dict[str, Any], captured: dict[str, Any]) -> list[str]:
    """Dotted paths where two response bodies disagree in shape or value.

    Compares keys structurally rather than diffing text, so "the usage block is
    named differently" reads as one finding rather than as every line changing.
    """
    found: list[str] = []

    def _walk(left: object, right: object, path: str) -> None:
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) | set(right)):
                where = f"{path}.{key}" if path else key
                if key not in left:
                    found.append(f"{where} (only in captured)")
                elif key not in right:
                    found.append(f"{where} (only in constructed)")
                else:
                    _walk(left[key], right[key], where)
            return
        if isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                found.append(f"{path} (length {len(left)} vs {len(right)})")
            for index, (a, b) in enumerate(zip(left, right, strict=False)):
                _walk(a, b, f"{path}[{index}]")
            return
        if type(left) is not type(right):
            found.append(f"{path} (type {type(left).__name__} vs {type(right).__name__})")
        elif left != right:
            found.append(path or "<root>")

    _walk(constructed, captured, "")
    return found
