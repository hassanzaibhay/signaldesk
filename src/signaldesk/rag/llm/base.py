"""What every provider looks like from the router's side.

The router knows four things about a provider: what it is, whether it can be
asked, what it costs to reach, and how to ask it. Everything provider-specific -
request shape, where the text lives in the response, how a rate limit announces
itself, what a safety block looks like - stays behind this interface.

Providers raise rather than returning sentinels. A rate limit is
``RateLimitError``, an unusable status is ``ProviderError``, and a transport
failure is whatever ``httpx`` raised after ``core/http.py`` exhausted its
retries. The router maps each to an outcome and moves down the chain; nothing
here decides whether to fail over.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Protocol

import httpx

from signaldesk.core.config import Settings
from signaldesk.core.errors import ProviderError, RateLimitError
from signaldesk.core.http import request
from signaldesk.core.logging import get_logger
from signaldesk.rag.llm.types import ModelIdentity, Refusal, TokenUsage

log = get_logger(__name__)

#: Sent to every provider. Deterministic output is worth more here than variety:
#: these calls are adjudications, and a cassette recorded at temperature 1 would
#: not describe what the next call does.
DEFAULT_TEMPERATURE = 0.0


@dataclass(frozen=True, slots=True)
class Message:
    """One turn. ``role`` is "system", "user" or "assistant"."""

    role: str
    content: str


@dataclass(frozen=True, slots=True)
class ChatRequest:
    """What the router asks a provider for, before provider-specific shaping."""

    messages: tuple[Message, ...]
    #: The caller's Pydantic model rendered as JSON Schema. Providers that can
    #: constrain decoding use it; the response is validated against the model
    #: either way, because a provider claiming schema support is not evidence
    #: that it honoured it.
    json_schema: dict[str, Any]
    schema_name: str
    #: Carried on the request because it is part of what identifies this
    #: interaction, which is what a cassette is keyed on.
    prompt_version: str = ""
    temperature: float = DEFAULT_TEMPERATURE
    max_output_tokens: int = 2048


@dataclass(frozen=True, slots=True)
class RawCompletion:
    """What a provider got back, before schema validation.

    ``refusal`` is set when the provider declined on its own safety grounds and
    returned no usable content. That is an answer and the router stops on it; it
    is not a reason to try the next provider.
    """

    text: str = ""
    usage: TokenUsage = field(default_factory=TokenUsage)
    refusal: Refusal | None = None
    #: The decoded response body, kept so a cassette records what arrived rather
    #: than what this code made of it.
    raw: dict[str, Any] = field(default_factory=dict)
    #: Set when this answer came off disk, so the provenance record can say so.
    cassette_key: str | None = None


class Provider(Protocol):
    """The whole of what the router needs."""

    name: str
    model: str
    endpoint: str

    @property
    def identity(self) -> ModelIdentity: ...

    @property
    def is_configured(self) -> bool:
        """Whether this provider can be asked at all.

        False means skipped, never an error: continuous integration runs with no
        keys and must still walk the chain.
        """
        ...

    def complete(
        self,
        chat: ChatRequest,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> RawCompletion: ...


class HttpProvider:
    """Shared plumbing for the three providers that are remote HTTP APIs.

    Subclasses supply the URL, the headers, the request body and the response
    parsing. This holds the one thing they must not each reinvent: the call goes
    through ``core/http.py``, so the timeouts, the tenacity policy and the
    ``Retry-After`` handling are the project's single retry philosophy rather
    than a second one per provider.
    """

    name: str = ""
    endpoint: str = ""

    def __init__(self, model: str, settings: Settings) -> None:
        self.model = model
        self.settings = settings

    @property
    def identity(self) -> ModelIdentity:
        return ModelIdentity(provider=self.name, model=self.model)

    @property
    def is_configured(self) -> bool:  # pragma: no cover - overridden by every subclass
        raise NotImplementedError

    def _url(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def _headers(self) -> dict[str, str]:  # pragma: no cover - overridden
        raise NotImplementedError

    def _body(self, chat: ChatRequest) -> dict[str, Any]:  # pragma: no cover - overridden
        raise NotImplementedError

    def _parse(self, payload: dict[str, Any]) -> RawCompletion:  # pragma: no cover - overridden
        raise NotImplementedError

    def complete(
        self,
        chat: ChatRequest,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> RawCompletion:
        """Ask, and translate the answer into the router's vocabulary.

        Cassette resolution happens here rather than in the router because it is
        an HTTP concern: replay works by swapping the transport under this
        client, and a provider that makes no HTTP call has no cassette. The
        router stayed ignorant of it for exactly that reason.
        """
        from signaldesk.core.http import build_client
        from signaldesk.rag.llm import cassettes

        cassette_key: str | None = None
        if transport is None and cassettes.current_mode() is cassettes.Mode.REPLAY:
            cassette_key = cassettes.key_for(self.name, self.model, chat.prompt_version, chat)
            transport = cassettes.replay_transport(cassettes.load(cassette_key))

        client = build_client(self.settings, transport=transport)
        try:
            response = request(
                "POST",
                self._url(),
                client=client,
                headers=self._headers(),
                body=self._body(chat),
                settings=self.settings,
                use_cache=False,
            )
        finally:
            client.close()

        if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
            message = f"{self.name} rate limited this call: {response.text[:200]}"
            raise RateLimitError(message)
        if response.status_code != httpx.codes.OK:
            message = (
                f"{self.name} returned {response.status_code} for model "
                f"{self.model}: {response.text[:200]}"
            )
            raise ProviderError(message)

        try:
            payload: dict[str, Any] = response.json()
        except ValueError as error:
            message = f"{self.name} returned a body that is not JSON: {response.text[:200]}"
            raise ProviderError(message) from error

        parsed = self._parse(payload)
        return replace(parsed, cassette_key=cassette_key)


def usage_from(payload: dict[str, Any], *, prompt: str, completion: str, total: str) -> TokenUsage:
    """Pull token counts out of a provider's usage block.

    Every field stays ``None`` when the provider did not report it. A zero would
    be indistinguishable from a real zero and would quietly become a fabricated
    measurement once these are summed.
    """

    def _int(value: object) -> int | None:
        return int(value) if isinstance(value, (int, float)) else None

    return TokenUsage(
        prompt=_int(payload.get(prompt)),
        completion=_int(payload.get(completion)),
        total=_int(payload.get(total)),
    )
