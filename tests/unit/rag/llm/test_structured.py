"""Parsing a typed object out of whatever the model actually said."""

from __future__ import annotations

import json

import pytest
from fakes import Verdict

from signaldesk.core.errors import StructuredOutputError
from signaldesk.rag.llm import structured
from signaldesk.rag.llm.base import Message

pytestmark = pytest.mark.unit

GOOD = json.dumps({"label": "labelled", "confident": True})


class TestParsing:
    def test_clean_json_validates(self) -> None:
        assert structured.parse(GOOD, Verdict) == Verdict(label="labelled", confident=True)

    @pytest.mark.parametrize(
        "wrapped",
        [
            f"```json\n{GOOD}\n```",
            f"```\n{GOOD}\n```",
            f"  ```json\n{GOOD}\n```  ",
        ],
    )
    def test_a_markdown_fence_is_stripped(self, wrapped: str) -> None:
        """Models fence JSON often enough that this is parsing, not a workaround."""
        assert structured.parse(wrapped, Verdict).label == "labelled"

    def test_surrounding_whitespace_is_tolerated(self) -> None:
        assert structured.parse(f"\n\n{GOOD}\n", Verdict).confident is True

    def test_prose_is_rejected(self) -> None:
        with pytest.raises(StructuredOutputError, match="not valid JSON"):
            structured.parse("Sure! Here is the answer.", Verdict)

    def test_empty_output_is_rejected(self) -> None:
        with pytest.raises(StructuredOutputError, match="no content"):
            structured.parse("   ", Verdict)

    def test_a_missing_field_names_the_field(self) -> None:
        with pytest.raises(StructuredOutputError) as caught:
            structured.parse('{"label": "x"}', Verdict)
        assert "confident" in str(caught.value)

    def test_a_wrong_type_is_rejected(self) -> None:
        """A provider that ignores the schema tends to stringify everything."""
        with pytest.raises(StructuredOutputError):
            structured.parse('{"label": "x", "confident": "yes please"}', Verdict)

    def test_the_error_names_the_schema(self) -> None:
        with pytest.raises(StructuredOutputError, match="Verdict"):
            structured.parse('{"label": "x"}', Verdict)


class TestTheSchema:
    def test_it_is_json_schema_for_the_model(self) -> None:
        schema = structured.json_schema_for(Verdict)
        assert set(schema["properties"]) == {"label", "confident"}
        assert schema["properties"]["confident"]["type"] == "boolean"

    def test_it_is_json_serialisable(self) -> None:
        """It is sent in a request body, so it has to survive serialisation."""
        json.dumps(structured.json_schema_for(Verdict))


class TestTheRepairTurn:
    def test_it_keeps_the_original_conversation(self) -> None:
        original = (Message(role="user", content="the question"),)
        repaired = structured.repair_messages(original, "bad", "why", Verdict)
        assert repaired[0] == original[0]

    def test_it_replays_the_rejected_text_as_the_assistant_turn(self) -> None:
        """The model is correcting a conversation, not being told about one."""
        repaired = structured.repair_messages((), "that was wrong", "why", Verdict)
        assert repaired[-2].role == "assistant"
        assert repaired[-2].content == "that was wrong"

    def test_it_states_the_problem_and_the_schema(self) -> None:
        repaired = structured.repair_messages((), "bad", "confident: field required", Verdict)
        instruction = repaired[-1].content
        assert "confident: field required" in instruction
        assert "confident" in instruction
        assert "no code fence" in instruction

    def test_a_very_long_rejected_body_is_truncated(self) -> None:
        """The repair turn must not blow the context budget on the bad answer."""
        repaired = structured.repair_messages((), "x" * 10_000, "why", Verdict)
        assert len(repaired[-2].content) == 4000
