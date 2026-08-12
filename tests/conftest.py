"""Shared test fixtures.

Unit tests construct settings explicitly with ``_env_file=None`` so that a local
``.env`` cannot change what they assert.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from signaldesk.core.config import Settings

SettingsFactory = Callable[..., Settings]


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every settings key from the environment.

    A test that asserts a default has to run without one, and the container
    exports most of these from its .env file.
    """
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)


@pytest.fixture
def make_settings(tmp_path: Path) -> SettingsFactory:
    """Build a Settings object isolated from the environment and the filesystem."""

    def factory(**overrides: object) -> Settings:
        values: dict[str, object] = {
            "django_secret_key": "test-only-not-a-secret",
            "data_dir": tmp_path / "data",
            "cache_dir": tmp_path / "cache",
            "model_dir": tmp_path / "models",
        }
        values.update(overrides)
        return Settings(_env_file=None, **values)

    return factory


@pytest.fixture
def settings(make_settings: SettingsFactory) -> Settings:
    """A ready-made isolated Settings object."""
    return make_settings()
