"""The one interaction the suite replays, defined once.

Both the tests and the recorder need to agree on exactly what is asked, or a
recording run refreshes cassettes the suite does not use and the suite replays
cassettes nobody re-records. Defining the probe here, in the package rather than
in the tests, is what keeps them in step.

It is deliberately trivial. This layer is being tested for routing, validation
and provenance, not for whether a model can answer a hard question, and a
trivial probe is one whose correct answer does not change when a provider
updates its weights.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from signaldesk.rag.llm.base import Message

PROMPT_VERSION = "router_probe_v1"
PROMPT_NAME = "router_probe"


class RouterProbe(BaseModel):
    """The smallest structured answer that still exercises typing and validation."""

    drug: str = Field(description="The drug named in the question.")
    reaction: str = Field(description="The adverse reaction named in the question.")
    #: Present so the schema has a non-string field: providers that ignore the
    #: schema tend to return every value as a string, and that is worth catching.
    terms_found: int = Field(description="How many of the two terms were present.")


def probe_messages() -> list[Message]:
    """The probe, rendered."""
    return [
        Message(
            role="system",
            content=(
                "You extract terms from pharmacovigilance text. Answer only with "
                "JSON matching the requested schema. Disproportionality findings "
                "are hypothesis-generating and never establish causation."
            ),
        ),
        Message(
            role="user",
            content=(
                "Sentence: 'Cases of rhabdomyolysis have been reported in patients "
                "taking atorvastatin.'\n"
                "Return the drug, the reaction, and how many of the two you found."
            ),
        ),
    ]
