"""What a model call is, and what it leaves behind.

Nothing here performs I/O. These are the records every other module in this
package produces or consumes, and two of them exist to satisfy project rules
rather than to make the code work:

* ``ModelRun`` is the provenance record. Every model-derived row in this project
  has to carry the model that produced it and the version of the prompt that
  asked, so the record is attached to every result from the start rather than
  bolted on when the first table that stores it appears.
* ``CallTrace`` is the failover record. When a chain of four providers produces
  one answer, the interesting information is what the other three did, and that
  has to survive the call rather than only reaching a log line.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import TypeVar

from pydantic import BaseModel, ConfigDict, Field

Schema = TypeVar("Schema", bound=BaseModel)


class ModelIdentity(BaseModel):
    """Which model, at which provider. The unit of "same model" comparisons."""

    model_config = ConfigDict(frozen=True)

    provider: str
    model: str

    def __str__(self) -> str:
        return f"{self.provider}/{self.model}"


class AttemptOutcome(StrEnum):
    """Why one provider did or did not produce the answer.

    The three ``skipped`` values are not failures. A provider with no API key is
    expected on a machine that has none, and CI has none at all; a provider that
    would have been the same model as the generator it is judging is skipped by
    design. Recording them distinctly is what lets the exhausted-chain error say
    whether anything was actually tried.
    """

    OK = "ok"
    REFUSED = "refused"
    SKIPPED_UNCONFIGURED = "skipped_unconfigured"
    SKIPPED_SAME_MODEL = "skipped_same_model"
    SKIPPED_NOT_ALLOWED = "skipped_not_allowed"
    TRANSPORT = "transport"
    RATE_LIMIT = "rate_limit"
    HTTP_STATUS = "http_status"
    SCHEMA = "schema"


#: Outcomes that mean the provider was actually asked and could not answer.
#: Distinguished from the skips so an exhausted chain can say which it was.
FAILURE_OUTCOMES = frozenset(
    {
        AttemptOutcome.TRANSPORT,
        AttemptOutcome.RATE_LIMIT,
        AttemptOutcome.HTTP_STATUS,
        AttemptOutcome.SCHEMA,
    }
)

#: Outcomes that end the walk. A refusal is an answer, not a failure.
TERMINAL_OUTCOMES = frozenset({AttemptOutcome.OK, AttemptOutcome.REFUSED})


class TokenUsage(BaseModel):
    """Tokens as the provider reported them.

    Every field is optional because not every provider reports usage, and a
    zero would be a fabricated measurement rather than a missing one.
    """

    model_config = ConfigDict(frozen=True)

    prompt: int | None = None
    completion: int | None = None
    total: int | None = None


class Attempt(BaseModel):
    """One provider's turn in the chain."""

    model_config = ConfigDict(frozen=True)

    identity: ModelIdentity
    outcome: AttemptOutcome
    #: Human-readable cause. Empty for the skips, which need no explanation
    #: beyond their outcome.
    detail: str = ""
    latency_ms: float = 0.0
    #: Structured-output repairs spent inside this provider. At most one.
    repairs: int = 0

    def __str__(self) -> str:
        suffix = f": {self.detail}" if self.detail else ""
        return f"{self.identity} {self.outcome}{suffix}"


class CallTrace(BaseModel):
    """Everything the chain did, in order."""

    model_config = ConfigDict(frozen=True)

    attempts: tuple[Attempt, ...] = ()

    @property
    def nothing_configured(self) -> bool:
        """Whether no provider was in a state to be asked.

        True when every entry is a skip. That is a deployment problem - no keys,
        or a chain that names providers none of which are enabled - and it reads
        very differently from four providers that were asked and failed.
        """
        return bool(self.attempts) and all(
            attempt.outcome
            in {
                AttemptOutcome.SKIPPED_UNCONFIGURED,
                AttemptOutcome.SKIPPED_SAME_MODEL,
                AttemptOutcome.SKIPPED_NOT_ALLOWED,
            }
            for attempt in self.attempts
        )

    @property
    def tried(self) -> tuple[Attempt, ...]:
        """Attempts that reached a provider, skips excluded."""
        return tuple(a for a in self.attempts if a.outcome not in _SKIPS)

    @property
    def collided(self) -> tuple[Attempt, ...]:
        """Attempts skipped because they were the model being judged."""
        return tuple(a for a in self.attempts if a.outcome is AttemptOutcome.SKIPPED_SAME_MODEL)

    def render(self) -> str:
        """One line per provider, for an error message or a log."""
        return "; ".join(str(attempt) for attempt in self.attempts) or "no providers in the chain"


_SKIPS = frozenset(
    {
        AttemptOutcome.SKIPPED_UNCONFIGURED,
        AttemptOutcome.SKIPPED_SAME_MODEL,
        AttemptOutcome.SKIPPED_NOT_ALLOWED,
    }
)


class ModelRun(BaseModel):
    """The provenance of one model-derived result.

    Carried on every completion. Nothing persists it yet - the table arrives
    with the labeledness adjudicator - but the record is produced here so that
    the rule "every model-derived database row records its model and prompt
    version" is satisfied by construction rather than by remembering to add the
    fields when the table is written.

    ``prompt_digest`` is a hash of the rendered messages. The prompt version
    says which template was used; the digest says that these exact inputs
    produced this exact output, which is what makes a stored result
    reproducible.
    """

    model_config = ConfigDict(frozen=True)

    identity: ModelIdentity
    prompt_name: str
    prompt_version: str
    prompt_digest: str
    usage: TokenUsage = TokenUsage()
    latency_ms: float = 0.0
    #: Set when the response came from a cassette rather than a provider.
    cassette_key: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))


class Refusal(BaseModel):
    """A provider declining to answer, which is a result and not a failure.

    A model that will not answer has told the caller something, and the caller
    decides what that means. Failing over to the next provider to shop for a
    more compliant one would be both wasteful and wrong.
    """

    model_config = ConfigDict(frozen=True)

    reason: str
    #: The provider's own code, verbatim, so the classification is auditable.
    provider_code: str = ""


@dataclass(frozen=True, slots=True)
class Completion[Value: BaseModel]:
    """What a successful call returns. Exactly one of value or refusal is set."""

    value: Value | None
    refusal: Refusal | None
    run: ModelRun
    trace: CallTrace

    def __post_init__(self) -> None:
        if (self.value is None) == (self.refusal is None):
            message = "a completion carries exactly one of value or refusal"
            raise ValueError(message)

    @property
    def refused(self) -> bool:
        return self.refusal is not None

    def unwrap(self) -> Value:
        """The value, or raise if this was a refusal.

        For callers that treat a refusal as unusable. Callers that handle
        refusals read ``value`` and ``refusal`` directly.
        """
        if self.value is None:
            message = f"the model refused: {self.refusal.reason if self.refusal else 'unknown'}"
            raise ValueError(message)
        return self.value
