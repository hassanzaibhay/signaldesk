"""Chain assembly, and the allowlist that keeps the bill at zero."""

from __future__ import annotations

from pathlib import Path

import pytest

from signaldesk.core.config import Settings
from signaldesk.rag.llm import registry
from signaldesk.rag.llm.errors import ProviderNotAllowedError

pytestmark = pytest.mark.unit


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    fields: dict[str, object] = {
        "django_secret_key": "test",
        "data_dir": tmp_path,
        "cache_dir": tmp_path / "cache",
    }
    fields.update(overrides)
    return Settings(**fields)  # type: ignore[arg-type]


class TestChainOrder:
    def test_the_generation_chain_follows_the_configured_order(self, tmp_path: Path) -> None:
        chain = registry.generation_chain(
            _settings(tmp_path, llm_provider_chain="gemini,groq,cerebras,ollama")
        )
        assert [p.name for p in chain] == ["gemini", "groq", "cerebras", "ollama"]

    def test_the_judge_chain_puts_the_configured_judge_first(self, tmp_path: Path) -> None:
        chain = registry.judge_chain(
            _settings(
                tmp_path, llm_provider_chain="gemini,groq,cerebras", llm_judge_provider="cerebras"
            )
        )
        assert next(p.name for p in chain) == "cerebras"

    def test_the_judge_chain_keeps_the_rest_as_failover(self, tmp_path: Path) -> None:
        """A judge whose provider is rate limited should still get an answer."""
        chain = registry.judge_chain(
            _settings(
                tmp_path, llm_provider_chain="gemini,groq,cerebras", llm_judge_provider="groq"
            )
        )
        assert [p.name for p in chain] == ["groq", "gemini", "cerebras"]

    def test_an_unknown_provider_in_the_chain_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ProviderNotAllowedError, match="openai"):
            registry.generation_chain(_settings(tmp_path, llm_provider_chain="openai"))


class TestConfiguredness:
    def test_a_provider_without_its_key_is_not_configured(self, tmp_path: Path) -> None:
        chain = registry.generation_chain(
            _settings(
                tmp_path, llm_provider_chain="gemini,groq", gemini_api_key="", groq_api_key=""
            )
        )
        assert not any(p.is_configured for p in chain)

    def test_a_provider_with_its_key_is_configured(self, tmp_path: Path) -> None:
        chain = registry.generation_chain(
            _settings(tmp_path, llm_provider_chain="groq", groq_api_key="k")
        )
        assert chain[0].is_configured

    def test_ollama_needs_an_explicit_enable_not_a_url(self, tmp_path: Path) -> None:
        """A URL is always present, so a URL-based check reports configured
        everywhere and turns every exhausted chain into a connection timeout."""
        off = registry.generation_chain(_settings(tmp_path, llm_provider_chain="ollama"))
        assert not off[0].is_configured

        on = registry.generation_chain(
            _settings(tmp_path, llm_provider_chain="ollama", ollama_enabled=True)
        )
        assert on[0].is_configured

    def test_ollama_is_off_by_default(self, tmp_path: Path) -> None:
        assert _settings(tmp_path).ollama_enabled is False


class TestTheFreeTierAllowlist:
    def test_a_paid_model_at_its_own_provider_is_refused_at_construction(
        self, tmp_path: Path
    ) -> None:
        """The model is claimed by groq but is not on its free list."""
        with pytest.raises(ProviderNotAllowedError, match="free-tier allowlist"):
            registry.build(
                "groq",
                preferred_model="llama-3.3-70b-versatile-paid",
                settings=_settings(tmp_path),
            )

    def test_a_model_belonging_to_another_provider_falls_back_to_the_default(
        self, tmp_path: Path
    ) -> None:
        """One setting cannot name a model for four providers, so this is not an
        error - the provider uses its own free-tier default."""
        provider = registry.build(
            "groq", preferred_model="gemini-2.5-flash", settings=_settings(tmp_path)
        )
        assert provider.model == "llama-3.3-70b-versatile"

    def test_an_allowed_model_is_used_as_asked(self, tmp_path: Path) -> None:
        provider = registry.build(
            "groq", preferred_model="llama-3.1-8b-instant", settings=_settings(tmp_path)
        )
        assert provider.model == "llama-3.1-8b-instant"

    def test_a_remote_ollama_is_refused(self, tmp_path: Path) -> None:
        """Ollama has no key, so a hosted URL is the way a bill could sneak in."""
        with pytest.raises(ProviderNotAllowedError, match="not on the free-tier allowlist"):
            registry.build(
                "ollama",
                preferred_model="",
                settings=_settings(tmp_path, ollama_base_url="https://ollama.paid-host.com"),
            )

    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost:11434",
            "http://127.0.0.1:11434",
            "http://ollama:11434",
            "http://host.docker.internal:11434",
        ],
    )
    def test_local_ollama_urls_are_allowed(self, tmp_path: Path, url: str) -> None:
        provider = registry.build(
            "ollama", preferred_model="", settings=_settings(tmp_path, ollama_base_url=url)
        )
        assert provider.name == "ollama"

    def test_every_default_model_is_on_its_own_allowlist(self) -> None:
        """A default outside its own free tier would be unreachable by design."""
        for name, (_cls, allowed, default) in registry.REGISTRY.items():
            if name == "ollama":
                continue
            assert default in allowed, f"{name} default {default} is not allowlisted"

    def test_no_two_providers_claim_the_same_model_name(self) -> None:
        """The fallback logic reads 'which provider claims this model', so an
        overlap would make that answer ambiguous."""
        seen: dict[str, str] = {}
        for name, (_cls, allowed, _default) in registry.REGISTRY.items():
            for model in allowed:
                assert model not in seen, f"{model} claimed by {seen.get(model)} and {name}"
                seen[model] = name
