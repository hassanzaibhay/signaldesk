"""Assembling the ordered chain, and the allowlist that keeps it free.

Total infrastructure cost is zero. The way that survives configuration drift is
that a provider pointed at something outside the free tier cannot be built at
all: ``ProviderNotAllowedError`` is raised here, at construction, before any
request is shaped. Editing an environment variable to name a paid model or a
hosted Ollama gets a startup failure, not a bill.

Two chains come out of this module. The generation chain is
``LLM_PROVIDER_CHAIN`` in order. The judge chain is the same providers with the
configured judge provider moved to the front, because a judge that has fallen
over should still fail over rather than give up - it just must not land on the
model it is grading, and that is the router's business rather than this one's.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from signaldesk.core.config import Settings, get_settings
from signaldesk.core.logging import get_logger
from signaldesk.rag.llm.base import Provider
from signaldesk.rag.llm.errors import ProviderNotAllowedError
from signaldesk.rag.llm.providers import cerebras, gemini, groq, ollama

log = get_logger(__name__)

#: Provider name to (class, allowed models, default model). The allowed sets are
#: free tiers as published. A model absent from its provider's set is refused.
REGISTRY: Final[dict[str, tuple[type[Provider], frozenset[str], str]]] = {
    "gemini": (gemini.GeminiProvider, gemini.ALLOWED_MODELS, gemini.DEFAULT_MODEL),
    "groq": (groq.GroqProvider, groq.ALLOWED_MODELS, groq.DEFAULT_MODEL),
    "cerebras": (cerebras.CerebrasProvider, cerebras.ALLOWED_MODELS, cerebras.DEFAULT_MODEL),
    "ollama": (ollama.OllamaProvider, frozenset(), ollama.DEFAULT_MODEL),
}


def _resolve_model(provider: str, preferred: str, allowed: frozenset[str], default: str) -> str:
    """The model to use, preferring the configured one when it is allowed.

    Three cases, and only the last is an error:

    * the preferred model is on this provider's free list, so use it;
    * it is on a *different* provider's free list, which is not a mistake - the
      chain names four providers and one setting cannot name a model for all of
      them - so this provider uses its own default;
    * it is on nobody's free list. That is either a typo or a paid model, and
      both should stop rather than silently become something else. Falling back
      here would mean the operator asked for one model and got another without
      being told.
    """
    if provider == "ollama":
        # Ollama runs whatever has been pulled locally; there is no vendor list
        # to check against and nothing to bill. The setting wins.
        return preferred or default
    if not preferred or preferred in allowed:
        return preferred or default
    if _claimed_by(preferred) is None:
        raise ProviderNotAllowedError(provider, model=preferred)
    return default


def _claimed_by(model: str) -> str | None:
    """Which provider's free tier lists this model, if any."""
    for name, (_cls, allowed, _default) in REGISTRY.items():
        if model in allowed:
            return name
    return None


def build(name: str, *, preferred_model: str, settings: Settings) -> Provider:
    """Construct one provider, or refuse.

    Raises ``ProviderNotAllowedError`` for an unknown provider, a model outside
    the provider's free tier, or an Ollama base URL that is not local.
    """
    entry = REGISTRY.get(name)
    if entry is None:
        raise ProviderNotAllowedError(name, model=preferred_model)

    provider_cls, allowed, default = entry
    model = _resolve_model(name, preferred_model, allowed, default)

    if name == "ollama" and not ollama.host_is_local(settings.ollama_base_url):
        raise ProviderNotAllowedError(name, endpoint=settings.ollama_base_url)

    if model != preferred_model and preferred_model:
        log.debug(
            "llm.registry.model_substituted",
            provider=name,
            requested=preferred_model,
            using=model,
        )
    return provider_cls(model, settings)  # type: ignore[call-arg]


def generation_chain(settings: Settings | None = None) -> tuple[Provider, ...]:
    """Providers in failover order for a generation call."""
    settings = settings or get_settings()
    return _chain(settings.provider_chain, settings.llm_generation_model, settings)


def judge_chain(settings: Settings | None = None) -> tuple[Provider, ...]:
    """Providers in failover order for a judge call.

    The configured judge provider goes first and the rest of the chain follows,
    so a judge whose preferred provider is rate limited still gets an answer
    from a different model rather than failing outright.
    """
    settings = settings or get_settings()
    order = list(settings.provider_chain)
    preferred = settings.llm_judge_provider
    if preferred in order:
        order.remove(preferred)
        order.insert(0, preferred)
    elif preferred:
        order.insert(0, preferred)
    return _chain(tuple(order), settings.llm_judge_model, settings)


def _chain(names: Sequence[str], preferred_model: str, settings: Settings) -> tuple[Provider, ...]:
    providers: list[Provider] = []
    for name in names:
        if name not in REGISTRY:
            raise ProviderNotAllowedError(name, model=preferred_model)
        providers.append(build(name, preferred_model=preferred_model, settings=settings))
    return tuple(providers)
