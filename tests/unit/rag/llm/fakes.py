"""Test doubles for the model layer.

A separate module rather than ``conftest.py`` because these are imported by
name, not injected as fixtures: the router tests build a different chain in
almost every case, so a factory fixture would be more ceremony than the thing it
replaced. pytest puts this directory on the path, so a plain import works.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from signaldesk.rag.llm.base import ChatRequest, RawCompletion
from signaldesk.rag.llm.types import ModelIdentity, Refusal, TokenUsage


class Verdict(BaseModel):
    """A small schema with a non-string field, to catch providers that stringify."""

    label: str
    confident: bool


class FakeProvider:
    """A provider that does exactly what a test tells it to.

    Stands in for the four real ones in the router and judge tests, so those
    tests describe routing rather than any vendor's JSON. The vendors' JSON is
    covered by the provider tests, against cassettes.
    """

    def __init__(
        self,
        name: str,
        model: str = "m",
        *,
        configured: bool = True,
        raises: Exception | None = None,
        text: str | None = None,
        refusal: Refusal | None = None,
        texts: list[str] | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self.endpoint = f"https://{name}.example"
        self._configured = configured
        self._raises = raises
        self._text = text
        self._refusal = refusal
        self._texts = texts
        self.calls: list[ChatRequest] = []

    @property
    def identity(self) -> ModelIdentity:
        return ModelIdentity(provider=self.name, model=self.model)

    @property
    def is_configured(self) -> bool:
        return self._configured

    def complete(self, chat: ChatRequest, *, transport: Any = None) -> RawCompletion:
        self.calls.append(chat)
        if self._raises is not None:
            raise self._raises
        if self._refusal is not None:
            return RawCompletion(refusal=self._refusal, usage=TokenUsage(total=3))
        if self._texts is not None:
            return RawCompletion(text=self._texts[len(self.calls) - 1], usage=TokenUsage(total=7))
        return RawCompletion(text=self._text or "", usage=TokenUsage(total=7))
