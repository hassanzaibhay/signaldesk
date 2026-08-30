"""Getting a typed object back, or nothing.

The caller declares a Pydantic model and receives an instance of it. There is no
path in this package that returns a raw string for the caller to parse: a
string that has to be parsed downstream is a schema violation that has not been
noticed yet.

Providers are asked to constrain their decoding where they can, but the response
is validated against the caller's model in every case. A provider advertising
structured output is not evidence that it honoured it, and two of the four in
this chain will happily return prose wrapped in a code fence.

One repair attempt is made against the same provider, handing back the
validation errors. If the repair also fails, this provider is finished and the
router moves on. That single attempt is what ``StructuredOutputError`` in
``core/errors.py`` already describes.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ValidationError

from signaldesk.core.errors import StructuredOutputError
from signaldesk.core.logging import get_logger
from signaldesk.rag.llm.base import Message

log = get_logger(__name__)

#: Models fence JSON in markdown often enough that stripping it is part of
#: parsing rather than a workaround. Non-greedy so a fence inside a string value
#: does not swallow the rest of the document.
_FENCE = re.compile(r"^\s*```(?:json)?\s*(?P<body>.*?)\s*```\s*$", re.DOTALL)


def json_schema_for(model: type[BaseModel]) -> dict[str, Any]:
    """The caller's model as JSON Schema, with references inlined.

    Providers vary in how much of JSON Schema they accept and several reject
    ``$ref`` outright, so the schema is emitted in inline mode. It is a hint to
    the provider; the authority is ``model_validate_json`` below.
    """
    return model.model_json_schema(ref_template="{model}")


def strip_fence(text: str) -> str:
    """Remove a surrounding markdown code fence, if there is one."""
    match = _FENCE.match(text)
    return match.group("body") if match else text.strip()


def parse(text: str, model: type[BaseModel]) -> BaseModel:
    """Validate ``text`` against ``model``, or raise ``StructuredOutputError``.

    The raised error carries the validation detail so the repair turn can hand
    the model its own mistakes rather than asking again and hoping.
    """
    candidate = strip_fence(text)
    if not candidate:
        message = "the model returned no content to validate"
        raise StructuredOutputError(message)
    try:
        return model.model_validate_json(candidate)
    except ValidationError as error:
        # Two different failures arrive as one exception type: output that is
        # not JSON at all, and JSON that does not fit the model. They are
        # separated because the repair turn says different things - one is "that
        # was not JSON", the other is "fix these fields" - and because a
        # provider that returns prose is failing differently from one that
        # returns the wrong shape. ValidationError subclasses ValueError, so a
        # second except clause would never be reached; the distinction has to be
        # made on the error type Pydantic reports.
        if any(item.get("type") == "json_invalid" for item in error.errors()):
            message = f"response was not valid JSON for {model.__name__}: {candidate[:200]}"
            raise StructuredOutputError(message) from error
        detail = _render_errors(error)
        message = f"response did not match {model.__name__}: {detail}"
        raise StructuredOutputError(message) from error


def _render_errors(error: ValidationError) -> str:
    parts = []
    for item in error.errors()[:8]:
        location = ".".join(str(piece) for piece in item.get("loc", ())) or "<root>"
        parts.append(f"{location}: {item.get('msg', 'invalid')}")
    return "; ".join(parts)


def repair_messages(
    original: tuple[Message, ...], bad_text: str, problem: str, model: type[BaseModel]
) -> tuple[Message, ...]:
    """The follow-up turn that asks the model to fix its own output.

    The rejected text is included as the assistant turn it actually was, so the
    model is correcting a conversation rather than being told about one.
    """
    schema = json.dumps(json_schema_for(model), indent=2, sort_keys=True)
    instruction = (
        f"That response was rejected: {problem}\n\n"
        f"Return only a JSON object valid against this schema, with no prose and "
        f"no code fence:\n{schema}"
    )
    return (
        *original,
        Message(role="assistant", content=bad_text[:4000]),
        Message(role="user", content=instruction),
    )
