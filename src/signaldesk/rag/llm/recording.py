"""Capturing real provider responses over the constructed ones.

Run by hand, by Hassan, with keys present. Never in continuous integration and
never automatically: this is the only code path in the project that makes a live
model call.

What it produces matters more than the cassettes themselves. Every cassette
committed today was written from vendor documentation, so the suite is currently
green against shapes nobody has verified. This run captures the real bodies,
writes the marker that promotes the still-constructed test from skipped to
enforced, and - the actual deliverable - reports every field where the captured
body differed from the hand-written one. That list is the evidence about which
parts of the router were built on a wrong assumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from signaldesk.core.config import Settings, get_settings
from signaldesk.core.errors import ProviderError
from signaldesk.core.logging import get_logger
from signaldesk.rag.llm import cassettes, registry, structured
from signaldesk.rag.llm.base import ChatRequest, Message, Provider

log = get_logger(__name__)


@dataclass(slots=True)
class RecordedOne:
    """What happened to one cassette."""

    key: str
    provider: str
    model: str
    was_constructed: bool
    differences: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def captured(self) -> bool:
        return not self.error


@dataclass(slots=True)
class RecordingReport:
    """What a whole recording run produced."""

    results: list[RecordedOne] = field(default_factory=list)
    marker: Path | None = None
    skipped_unconfigured: list[str] = field(default_factory=list)

    @property
    def captured(self) -> list[RecordedOne]:
        return [item for item in self.results if item.captured]

    @property
    def failed(self) -> list[RecordedOne]:
        return [item for item in self.results if not item.captured]

    @property
    def changed(self) -> list[RecordedOne]:
        return [item for item in self.captured if item.differences]


def record(
    interactions: list[tuple[str, ChatRequest]],
    *,
    settings: Settings | None = None,
    root: Path | None = None,
) -> RecordingReport:
    """Capture every interaction against every configured provider.

    ``interactions`` is a list of ``(prompt_version, request)``. Each is sent to
    each configured provider in the chain, because a cassette is keyed on the
    provider as well as the prompt and the router needs one per provider it
    might reach.
    """
    settings = settings or get_settings()
    report = RecordingReport()

    for provider in registry.generation_chain(settings):
        if not provider.is_configured:
            report.skipped_unconfigured.append(provider.name)
            log.info("llm.record.skipped", provider=provider.name)
            continue
        for prompt_version, chat in interactions:
            report.results.append(_record_one(provider, chat, prompt_version, root=root))

    captured = [item.key for item in report.captured]
    if captured:
        report.marker = cassettes.write_marker(
            captured,
            {item.key: item.differences for item in report.captured if item.differences},
            root=root,
        )
    return report


def _record_one(
    provider: Provider, chat: ChatRequest, prompt_version: str, *, root: Path | None
) -> RecordedOne:
    key = cassettes.key_for(provider.name, provider.model, prompt_version, chat)
    result = RecordedOne(
        key=key, provider=provider.name, model=provider.model, was_constructed=False
    )

    previous = None
    try:
        previous = cassettes.load(key, root)
        result.was_constructed = previous.constructed
    except ProviderError:
        # No cassette yet. Recording one is the point; nothing to compare to.
        pass

    try:
        # transport=None, so this is the live call. The only one in the project.
        raw = provider.complete(chat, transport=None)
    except Exception as error:
        # Broad on purpose. One provider answering strangely must not abandon
        # a recording run over every other provider and interaction; the
        # failure is recorded against this cassette and the run continues.
        result.error = repr(error)[:300]
        log.error("llm.record.failed", provider=provider.name, key=key, error=result.error)
        return result

    if previous is not None:
        result.differences = cassettes.differing_fields(previous.body, raw.raw)

    cassettes.save(
        cassettes.Cassette(
            key=key,
            provider=provider.name,
            model=provider.model,
            prompt_version=prompt_version,
            schema_name=chat.schema_name,
            status_code=200,
            body=raw.raw,
            constructed=False,
            constructed_at=previous.constructed_at if previous else "",
            recorded_at=datetime.now(tz=UTC).isoformat(),
        ),
        root,
    )
    log.info(
        "llm.record.captured",
        provider=provider.name,
        key=key,
        differences=len(result.differences),
    )
    return result


def default_interactions() -> list[tuple[str, ChatRequest]]:
    """The interactions the test suite replays, so a run refreshes exactly those.

    Kept beside the recorder rather than in the tests so that recording and
    replaying cannot drift apart: if the suite needs a new cassette, it is added
    here and the next recording run captures it.
    """
    from signaldesk.rag.llm.fixtures import RouterProbe, probe_messages

    return [
        (
            "router_probe_v1",
            ChatRequest(
                messages=tuple(probe_messages()),
                json_schema=structured.json_schema_for(RouterProbe),
                schema_name=RouterProbe.__name__,
            ),
        )
    ]


def render(report: RecordingReport) -> str:
    """The human summary printed by the command."""
    lines = [
        f"captured {len(report.captured)} cassette(s), {len(report.failed)} failed",
    ]
    if report.skipped_unconfigured:
        lines.append(f"skipped, not configured: {', '.join(report.skipped_unconfigured)}")
    for item in report.failed:
        lines.append(f"  FAILED {item.provider} {item.key}: {item.error}")
    if report.changed:
        lines.append("")
        lines.append("fields where the captured body differed from the constructed one:")
        for item in report.changed:
            lines.append(f"  {item.provider} {item.key}")
            for path in item.differences[:40]:
                lines.append(f"    {path}")
    elif report.captured:
        lines.append("no field differed from the constructed bodies")
    if report.marker:
        lines.append(f"marker written to {report.marker}")
    return "\n".join(lines)


def probe_message_list() -> list[Message]:
    """Re-exported for the command, so it does not import the fixtures module."""
    from signaldesk.rag.llm.fixtures import probe_messages

    return probe_messages()
