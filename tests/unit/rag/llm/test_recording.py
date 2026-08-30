"""The recorder, driven with fake providers so nothing goes to the network.

The recorder is the one code path that is allowed to make a live call, which is
exactly why its behaviour has to be pinned offline: nobody is going to run it
twice to check that it reports differences correctly.

What matters here is the reporting. Capturing a response is the easy half; the
half that earns the exercise is the list of fields where the captured body
differed from the hand-written one, because that list is the evidence about
which parts of the router were built on a wrong assumption.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fakes import FakeProvider

from signaldesk.core.config import Settings
from signaldesk.core.errors import ProviderError
from signaldesk.rag.llm import cassettes, recording, structured
from signaldesk.rag.llm.base import ChatRequest, Message, RawCompletion
from signaldesk.rag.llm.fixtures import RouterProbe

pytestmark = pytest.mark.unit


class RecordingProvider(FakeProvider):
    """A fake whose ``raw`` body is what a recorder would capture."""

    def __init__(self, name: str, body: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(name, **kwargs)
        self._body = body

    def complete(self, chat: ChatRequest, *, transport: Any = None) -> RawCompletion:
        self.calls.append(chat)
        if self._raises is not None:
            raise self._raises
        return RawCompletion(text="{}", raw=self._body)


def _chat() -> ChatRequest:
    return ChatRequest(
        messages=(Message(role="user", content="probe"),),
        json_schema=structured.json_schema_for(RouterProbe),
        schema_name="RouterProbe",
        prompt_version="v1",
    )


def _settings(tmp_path: Path) -> Settings:
    return Settings(django_secret_key="test", data_dir=tmp_path, cache_dir=tmp_path / "cache")


class TestRecordingOne:
    def test_a_captured_body_is_written_as_not_constructed(
        self, cassette_root: Path, tmp_path: Path
    ) -> None:
        provider = RecordingProvider("groq", {"choices": [{"message": {"content": "{}"}}]})
        result = recording._record_one(provider, _chat(), "v1", root=cassette_root)

        assert result.captured
        stored = cassettes.load(result.key, cassette_root)
        assert stored.constructed is False
        assert stored.recorded_at
        assert stored.body == {"choices": [{"message": {"content": "{}"}}]}

    def test_replacing_a_constructed_cassette_reports_the_fields_that_differed(
        self, cassette_root: Path
    ) -> None:
        """The point of the exercise. A renamed usage field must show up here."""
        chat = _chat()
        key = cassettes.key_for("groq", "m", "v1", chat)
        cassettes.save(
            cassettes.Cassette(
                key=key,
                provider="groq",
                model="m",
                prompt_version="v1",
                schema_name="RouterProbe",
                status_code=200,
                body={"usage": {"total": 5}},
                constructed=True,
                constructed_at="2026-08-30",
            ),
            cassette_root,
        )
        provider = RecordingProvider("groq", {"usage": {"total_tokens": 5}})
        result = recording._record_one(provider, chat, "v1", root=cassette_root)

        assert result.was_constructed
        assert "usage.total (only in constructed)" in result.differences
        assert "usage.total_tokens (only in captured)" in result.differences

    def test_an_identical_capture_reports_no_differences(self, cassette_root: Path) -> None:
        chat = _chat()
        key = cassettes.key_for("groq", "m", "v1", chat)
        body = {"choices": [{"message": {"content": "{}"}}]}
        cassettes.save(
            cassettes.Cassette(
                key=key,
                provider="groq",
                model="m",
                prompt_version="v1",
                schema_name="RouterProbe",
                status_code=200,
                body=body,
                constructed=True,
                constructed_at="2026-08-30",
            ),
            cassette_root,
        )
        result = recording._record_one(
            RecordingProvider("groq", body), chat, "v1", root=cassette_root
        )
        assert result.differences == []

    def test_a_provider_failure_is_recorded_and_does_not_raise(self, cassette_root: Path) -> None:
        """One provider answering strangely must not abandon the whole run."""
        provider = RecordingProvider("groq", {}, raises=ProviderError("500"))
        result = recording._record_one(provider, _chat(), "v1", root=cassette_root)
        assert not result.captured
        assert "500" in result.error


class TestRecordingARun:
    def test_unconfigured_providers_are_skipped_and_named(
        self, cassette_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        chain = (
            RecordingProvider("gemini", {"a": 1}, configured=False),
            RecordingProvider("groq", {"b": 2}),
        )
        monkeypatch.setattr(recording.registry, "generation_chain", lambda settings: chain)
        report = recording.record(
            [("v1", _chat())], settings=_settings(tmp_path), root=cassette_root
        )
        assert report.skipped_unconfigured == ["gemini"]
        assert len(report.captured) == 1

    def test_the_marker_is_written_and_lists_what_changed(
        self, cassette_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Its existence is what promotes the still-constructed test from
        skipped to enforced, so it must only appear after a real capture."""
        chain = (RecordingProvider("groq", {"b": 2}),)
        monkeypatch.setattr(recording.registry, "generation_chain", lambda settings: chain)

        assert not cassettes.has_been_recorded(cassette_root)
        report = recording.record(
            [("v1", _chat())], settings=_settings(tmp_path), root=cassette_root
        )

        assert report.marker is not None
        assert cassettes.has_been_recorded(cassette_root)
        payload = json.loads(report.marker.read_text(encoding="utf-8"))
        assert payload["keys"] == [report.captured[0].key]
        assert payload["recorded_at"]

    def test_no_marker_when_nothing_was_captured(
        self, cassette_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A run where every provider was unconfigured has proved nothing."""
        chain = (RecordingProvider("groq", {}, configured=False),)
        monkeypatch.setattr(recording.registry, "generation_chain", lambda settings: chain)
        report = recording.record(
            [("v1", _chat())], settings=_settings(tmp_path), root=cassette_root
        )
        assert report.marker is None
        assert not cassettes.has_been_recorded(cassette_root)


class TestTheSummary:
    def test_it_counts_captures_and_failures(self) -> None:
        report = recording.RecordingReport(
            results=[
                recording.RecordedOne("k1", "groq", "m", was_constructed=True),
                recording.RecordedOne("k2", "gemini", "m", was_constructed=True, error="boom"),
            ]
        )
        rendered = recording.render(report)
        assert "captured 1 cassette(s), 1 failed" in rendered
        assert "FAILED gemini k2: boom" in rendered

    def test_it_lists_the_differing_fields(self) -> None:
        report = recording.RecordingReport(
            results=[
                recording.RecordedOne(
                    "k1", "groq", "m", was_constructed=True, differences=["usage.total_tokens"]
                )
            ]
        )
        rendered = recording.render(report)
        assert "fields where the captured body differed" in rendered
        assert "usage.total_tokens" in rendered

    def test_it_says_so_when_nothing_differed(self) -> None:
        report = recording.RecordingReport(
            results=[recording.RecordedOne("k1", "groq", "m", was_constructed=True)]
        )
        assert "no field differed" in recording.render(report)

    def test_it_names_skipped_providers(self) -> None:
        report = recording.RecordingReport(skipped_unconfigured=["ollama"])
        assert "skipped, not configured: ollama" in recording.render(report)


def test_the_default_interactions_are_the_ones_the_suite_replays() -> None:
    """Recording and replaying must not drift apart."""
    interactions = recording.default_interactions()
    assert len(interactions) == 1
    version, chat = interactions[0]
    assert version == "router_probe_v1"
    assert chat.schema_name == "RouterProbe"

    # The key this produces must be one of the committed cassettes.
    committed = {c.key for c in cassettes.all_cassettes()}
    assert cassettes.key_for("groq", "llama-3.3-70b-versatile", version, chat) in committed
