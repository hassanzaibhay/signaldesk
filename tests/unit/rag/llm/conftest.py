"""Shared fixtures for the model layer.

The autouse fixture is the important one: it pins replay mode for every test in
this package. A developer with provider keys in their environment gets disk, the
same as continuous integration does, so a test that would only pass with a live
key cannot be written here by accident.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fakes import Verdict

from signaldesk.core.config import Settings
from signaldesk.rag.llm import cassettes, structured
from signaldesk.rag.llm.base import ChatRequest, Message


@pytest.fixture(autouse=True)
def _force_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(cassettes.MODE_ENV_VAR, str(cassettes.Mode.REPLAY))


@pytest.fixture
def no_keys(tmp_path: Path) -> Settings:
    """The environment continuous integration actually runs in."""
    return Settings(
        django_secret_key="test",
        data_dir=tmp_path,
        cache_dir=tmp_path / "cache",
        gemini_api_key="",
        groq_api_key="",
        cerebras_api_key="",
        ollama_enabled=False,
    )


@pytest.fixture
def all_keys(tmp_path: Path) -> Settings:
    return Settings(
        django_secret_key="test",
        data_dir=tmp_path,
        cache_dir=tmp_path / "cache",
        gemini_api_key="g",
        groq_api_key="q",
        cerebras_api_key="c",
        ollama_enabled=True,
    )


@pytest.fixture
def messages() -> list[Message]:
    return [Message(role="user", content="Is nausea labelled for atorvastatin?")]


@pytest.fixture
def chat(messages: list[Message]) -> ChatRequest:
    return ChatRequest(
        messages=tuple(messages),
        json_schema=structured.json_schema_for(Verdict),
        schema_name=Verdict.__name__,
    )


@pytest.fixture
def cassette_root(tmp_path: Path) -> Path:
    root = tmp_path / "cassettes"
    root.mkdir()
    return root
