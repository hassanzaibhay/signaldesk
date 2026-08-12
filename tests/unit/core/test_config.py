"""Settings behaviour: defaults, parsing, derived values, and hard failures."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from signaldesk.core.config import Settings, get_settings

pytestmark = pytest.mark.unit


def test_defaults_match_the_container_layout(settings: Settings) -> None:
    assert settings.postgres_host == "postgres"
    assert settings.postgres_port == 5432
    assert settings.faers_start_quarter == "2012Q4"
    assert settings.django_debug is False


def test_allowed_hosts_splits_the_comma_separated_value() -> None:
    settings = Settings(
        _env_file=None,
        django_secret_key="k",
        django_allowed_hosts="localhost, 127.0.0.1 ,web",
    )
    assert settings.allowed_hosts == ["localhost", "127.0.0.1", "web"]


def test_provider_chain_preserves_failover_order() -> None:
    settings = Settings(
        _env_file=None,
        django_secret_key="k",
        llm_provider_chain="gemini,groq,cerebras,ollama",
    )
    assert settings.provider_chain == ("gemini", "groq", "cerebras", "ollama")


def test_database_url_is_built_from_the_parts(settings: Settings) -> None:
    assert settings.database_url == (
        f"postgresql://{settings.postgres_user}:{settings.postgres_password}"
        f"@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
    )


def test_derived_paths_sit_under_the_configured_directories(settings: Settings) -> None:
    assert settings.http_cache_dir == settings.cache_dir / "http"
    assert settings.duckdb_path == settings.data_dir / "analytics.duckdb"
    assert isinstance(settings.data_dir, Path)


def test_environment_overrides_a_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_PORT", "6543")
    settings = Settings(_env_file=None, django_secret_key="k")
    assert settings.postgres_port == 6543


def test_secret_key_has_no_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DJANGO_SECRET_KEY", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_overlap_larger_than_the_chunk_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            django_secret_key="k",
            chunk_target_tokens=200,
            chunk_overlap_tokens=200,
        )


def test_non_positive_top_k_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, django_secret_key="k", dense_top_k=0)


def test_get_settings_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DJANGO_SECRET_KEY", "cached-key")
    get_settings.cache_clear()
    try:
        first = get_settings()
        assert get_settings() is first
    finally:
        get_settings.cache_clear()
