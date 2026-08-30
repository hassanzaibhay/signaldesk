"""Cassette keys, replay, and the marker that says whether any of this is real.

The last class here is the important one. Every cassette in this repository was
written from vendor documentation, so the rest of the suite is green against
shapes that have never been compared with a provider. That test is the thing
that will eventually say so out loud.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fakes import Verdict

from signaldesk.rag.llm import cassettes, structured
from signaldesk.rag.llm.base import ChatRequest, Message
from signaldesk.rag.llm.errors import CassetteMissError

pytestmark = pytest.mark.unit

#: Spelled from ordinals so this file cannot itself contain the sequence it
#: asserts against.
CRLF = bytes([13, 10])


def _chat(content: str = "hello", schema_name: str = "Verdict") -> ChatRequest:
    return ChatRequest(
        messages=(Message(role="user", content=content),),
        json_schema=structured.json_schema_for(Verdict),
        schema_name=schema_name,
        prompt_version="v1",
    )


def _cassette(key: str = "k", **overrides: object) -> cassettes.Cassette:
    fields: dict[str, object] = {
        "key": key,
        "provider": "groq",
        "model": "llama-3.3-70b-versatile",
        "prompt_version": "v1",
        "schema_name": "Verdict",
        "status_code": 200,
        "body": {"choices": [{"message": {"content": "{}"}}]},
        "constructed": True,
        "constructed_at": "2026-08-30",
    }
    fields.update(overrides)
    return cassettes.Cassette(**fields)  # type: ignore[arg-type]


class TestTheKey:
    def test_it_is_stable_for_the_same_interaction(self) -> None:
        first = cassettes.key_for("groq", "m", "v1", _chat())
        second = cassettes.key_for("groq", "m", "v1", _chat())
        assert first == second

    @pytest.mark.parametrize(
        ("provider", "model", "version", "content", "schema"),
        [
            ("gemini", "m", "v1", "hello", "Verdict"),
            ("groq", "other", "v1", "hello", "Verdict"),
            ("groq", "m", "v2", "hello", "Verdict"),
            ("groq", "m", "v1", "different", "Verdict"),
            ("groq", "m", "v1", "hello", "OtherSchema"),
        ],
    )
    def test_every_component_changes_it(
        self, provider: str, model: str, version: str, content: str, schema: str
    ) -> None:
        """All five identify the interaction, so all five must move the key."""
        base = cassettes.key_for("groq", "m", "v1", _chat())
        other = cassettes.key_for(provider, model, version, _chat(content, schema))
        assert other != base

    def test_it_is_a_filename(self) -> None:
        key = cassettes.key_for("groq", "m", "v1", _chat())
        assert key.isalnum()
        assert len(key) == 32


class TestReplay:
    def test_a_saved_cassette_round_trips(self, cassette_root: Path) -> None:
        cassettes.save(_cassette(), cassette_root)
        loaded = cassettes.load("k", cassette_root)
        assert loaded.provider == "groq"
        assert loaded.constructed is True
        assert loaded.body["choices"][0]["message"]["content"] == "{}"

    def test_a_miss_raises_and_never_falls_back_to_the_network(self, cassette_root: Path) -> None:
        """A silent fallback would pass on a machine with keys and fail in CI."""
        with pytest.raises(CassetteMissError) as caught:
            cassettes.load("absent", cassette_root)
        assert "absent" in str(caught.value)
        assert "record-cassettes" in str(caught.value)

    def test_an_unreadable_cassette_is_a_miss_not_a_crash(self, cassette_root: Path) -> None:
        (cassette_root / "broken.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(CassetteMissError, match="unreadable"):
            cassettes.load("broken", cassette_root)

    def test_the_transport_serves_the_recorded_body(self) -> None:
        transport = cassettes.replay_transport(_cassette(body={"ok": True}, status_code=201))
        client = httpx.Client(transport=transport)
        response = client.post("https://anything.example/v1", json={})
        assert response.status_code == 201
        assert response.json() == {"ok": True}

    def test_the_writer_pins_lf_regardless_of_platform(self, cassette_root: Path) -> None:
        """Cassettes are committed, and the portability gate rejects CRLF in
        tracked text. Python's text mode would write CRLF on Windows, so a
        recording run there would break the build it is meant to feed."""
        path = cassettes.save(_cassette(), cassette_root)
        assert CRLF not in path.read_bytes()

    def test_the_marker_also_pins_lf(self, cassette_root: Path) -> None:
        path = cassettes.write_marker(["k"], {}, cassette_root)
        assert CRLF not in path.read_bytes()

    def test_saved_json_is_sorted_and_newline_terminated(self, cassette_root: Path) -> None:
        path = cassettes.save(_cassette(), cassette_root)
        text = path.read_text(encoding="utf-8")
        assert text.endswith("\n")
        assert list(json.loads(text)) == sorted(json.loads(text))


class TestMode:
    def test_replay_is_the_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A developer with keys still gets disk."""
        monkeypatch.delenv(cassettes.MODE_ENV_VAR, raising=False)
        assert cassettes.current_mode() is cassettes.Mode.REPLAY

    @pytest.mark.parametrize("value", ["record", "RECORD", " off ", "replay"])
    def test_known_modes_are_accepted_case_insensitively(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv(cassettes.MODE_ENV_VAR, value)
        assert cassettes.current_mode() is cassettes.Mode(value.strip().lower())

    def test_an_unknown_mode_falls_back_to_replay(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The safe direction: a typo must not enable live calls."""
        monkeypatch.setenv(cassettes.MODE_ENV_VAR, "recrod")
        assert cassettes.current_mode() is cassettes.Mode.REPLAY


class TestDifferenceReporting:
    def test_identical_bodies_differ_nowhere(self) -> None:
        assert cassettes.differing_fields({"a": 1}, {"a": 1}) == []

    def test_a_renamed_field_is_reported_from_both_sides(self) -> None:
        found = cassettes.differing_fields({"usage": {"total": 1}}, {"usage": {"total_tokens": 1}})
        assert "usage.total (only in constructed)" in found
        assert "usage.total_tokens (only in captured)" in found

    def test_a_changed_type_is_reported_as_a_type(self) -> None:
        found = cassettes.differing_fields({"n": 1}, {"n": "1"})
        assert found == ["n (type int vs str)"]

    def test_nested_lists_are_compared_positionally(self) -> None:
        found = cassettes.differing_fields(
            {"choices": [{"finish_reason": "stop"}]},
            {"choices": [{"finish_reason": "length"}]},
        )
        assert found == ["choices[0].finish_reason"]

    def test_a_length_change_is_reported(self) -> None:
        found = cassettes.differing_fields({"c": [1]}, {"c": [1, 2]})
        assert "c (length 1 vs 2)" in found


class TestTheCommittedCassettesAreHonestlyLabelled:
    def test_every_committed_cassette_declares_whether_it_was_constructed(self) -> None:
        found = cassettes.all_cassettes()
        assert found, "the probe cassettes should be committed"
        for cassette in found:
            assert isinstance(cassette.constructed, bool)
            if cassette.constructed:
                assert cassette.constructed_at, f"{cassette.key} has no constructed date"

    def test_there_is_one_cassette_per_provider_in_the_chain(self) -> None:
        providers = {cassette.provider for cassette in cassettes.all_cassettes()}
        assert providers == {"gemini", "groq", "cerebras", "ollama"}

    @pytest.mark.skipif(
        not cassettes.has_been_recorded(),
        reason=(
            "no recording run has happened yet, so every cassette is still "
            "constructed from documentation. Run 'signaldesk evals "
            "record-cassettes' to capture real responses; until then this "
            "cannot pass, and skipping says so rather than reporting a green "
            "tick nobody has earned."
        ),
    )
    def test_no_cassette_is_still_constructed_after_a_recording_run(self) -> None:
        """The marker exists, so the hand-written bodies should all be gone."""
        stale = [c.key for c in cassettes.all_cassettes() if c.constructed]
        assert not stale, (
            f"a recording run has happened but these are still hand-written: {stale}. "
            "Re-record them or delete them."
        )
