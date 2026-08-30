"""The whole layer, offline, through the real providers and the real cassettes.

Everything above this file substitutes something: the router tests use fake
providers, the provider tests use hand-built transports. This one uses the real
registry, the real provider classes, the real committed cassettes, and the
project's own HTTP client - and asserts the property continuous integration
depends on, which is that the answer comes off disk with no key anywhere.

If this passes with the environment stripped of credentials, the suite cannot
have reached a provider.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from signaldesk.core.config import Settings
from signaldesk.rag.llm import registry, router
from signaldesk.rag.llm.errors import AllProvidersFailedError, CassetteMissError
from signaldesk.rag.llm.fixtures import PROMPT_NAME, PROMPT_VERSION, RouterProbe, probe_messages

pytestmark = pytest.mark.unit

#: Every credential the layer knows about. Cleared, not merely absent.
KEY_VARS = (
    "GEMINI_API_KEY",
    "GROQ_API_KEY",
    "CEREBRAS_API_KEY",
    "OLLAMA_ENABLED",
)


@pytest.fixture
def stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in KEY_VARS:
        monkeypatch.delenv(name, raising=False)


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    fields: dict[str, object] = {
        "django_secret_key": "test",
        "data_dir": tmp_path,
        "cache_dir": tmp_path / "cache",
        "gemini_api_key": "",
        "groq_api_key": "",
        "cerebras_api_key": "",
        "ollama_enabled": False,
    }
    fields.update(overrides)
    return Settings(**fields)  # type: ignore[arg-type]


def _run(settings: Settings):
    return router.complete(
        probe_messages(),
        schema=RouterProbe,
        prompt_name=PROMPT_NAME,
        prompt_version=PROMPT_VERSION,
        chain=registry.generation_chain(settings),
        settings=settings,
    )


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("gemini", "gemini-2.5-flash"),
        ("groq", "llama-3.3-70b-versatile"),
        ("cerebras", "llama-3.3-70b"),
        ("ollama", "qwen2.5:7b-instruct"),
    ],
)
def test_every_provider_answers_the_probe_from_its_cassette(
    stripped: None, tmp_path: Path, provider: str, model: str
) -> None:
    """One committed cassette per provider, replayed through its own parser."""
    settings = _settings(
        tmp_path,
        llm_provider_chain=provider,
        llm_generation_model=model,
        **({"ollama_enabled": True} if provider == "ollama" else {f"{provider}_api_key": "x"}),
    )
    result = _run(settings)

    assert result.value == RouterProbe(
        drug="atorvastatin", reaction="rhabdomyolysis", terms_found=2
    )
    assert result.run.identity.provider == provider
    assert result.run.identity.model == model
    assert result.run.cassette_key, "the answer must be recorded as coming off disk"
    assert result.run.usage.total is not None


def test_the_chain_fails_over_across_real_providers_on_disk(stripped: None, tmp_path: Path) -> None:
    """gemini has no key, so groq answers - and its cassette is the one used."""
    settings = _settings(
        tmp_path,
        llm_provider_chain="gemini,groq",
        llm_generation_model="llama-3.3-70b-versatile",
        groq_api_key="x",
    )
    result = _run(settings)
    assert result.run.identity.provider == "groq"
    assert result.trace.attempts[0].identity.provider == "gemini"


def test_with_no_credentials_at_all_the_chain_is_skipped_not_attempted(
    stripped: None, tmp_path: Path
) -> None:
    """The state continuous integration runs in. Nothing is asked, so nothing
    can reach the network, and the error says it was a configuration problem."""
    settings = _settings(tmp_path, llm_provider_chain="gemini,groq,cerebras,ollama")
    with pytest.raises(AllProvidersFailedError) as caught:
        _run(settings)

    assert caught.value.nothing_configured
    assert not caught.value.trace.tried
    assert len(caught.value.trace.attempts) == 4


def test_an_unrecorded_interaction_raises_rather_than_reaching_out(
    stripped: None, tmp_path: Path
) -> None:
    """Replay never falls back to a live call, so a new prompt is a loud miss."""
    settings = _settings(
        tmp_path,
        llm_provider_chain="groq",
        llm_generation_model="llama-3.3-70b-versatile",
        groq_api_key="x",
    )
    with pytest.raises(CassetteMissError):
        router.complete(
            probe_messages(),
            schema=RouterProbe,
            prompt_name=PROMPT_NAME,
            prompt_version="a-version-nobody-recorded",
            chain=registry.generation_chain(settings),
            settings=settings,
        )
