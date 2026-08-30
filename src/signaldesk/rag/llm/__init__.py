"""The model layer: ordered failover, typed output, provenance, cassettes.

Nothing else in this project talks to a model provider. Callers ask for a
Pydantic model and get an instance of it, or a refusal, or a typed error; they
never receive a string to parse and never choose a provider.

    from signaldesk.rag.llm import Message, complete

    result = complete(
        [Message(role="user", content=rendered)],
        schema=LabelednessVerdict,
        prompt_name="labeledness",
        prompt_version="labeledness_v1",
    )
    if result.refused:
        ...
    verdict = result.unwrap()
    provenance = result.run   # model identity and prompt version, for the row
"""

from __future__ import annotations

from signaldesk.rag.llm.base import ChatRequest, Message, Provider, RawCompletion
from signaldesk.rag.llm.errors import (
    AllProvidersFailedError,
    CassetteMissError,
    ProviderNotAllowedError,
    SameModelError,
)
from signaldesk.rag.llm.judge import judge
from signaldesk.rag.llm.registry import generation_chain, judge_chain
from signaldesk.rag.llm.router import complete
from signaldesk.rag.llm.types import (
    Attempt,
    AttemptOutcome,
    CallTrace,
    Completion,
    ModelIdentity,
    ModelRun,
    Refusal,
    TokenUsage,
)

__all__ = [
    "AllProvidersFailedError",
    "Attempt",
    "AttemptOutcome",
    "CallTrace",
    "CassetteMissError",
    "ChatRequest",
    "Completion",
    "Message",
    "ModelIdentity",
    "ModelRun",
    "Provider",
    "ProviderNotAllowedError",
    "RawCompletion",
    "Refusal",
    "SameModelError",
    "TokenUsage",
    "complete",
    "generation_chain",
    "judge",
    "judge_chain",
]
