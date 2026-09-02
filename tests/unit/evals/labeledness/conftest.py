"""Fixture wrappers over the builders in ``_builders``."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from _builders import build_manifest, build_screen

from signaldesk.evals.labeledness.manifest import SampleManifest, Screen


@pytest.fixture
def screen_factory() -> Callable[..., Screen]:
    return build_screen


@pytest.fixture
def manifest_factory() -> Callable[..., SampleManifest]:
    return build_manifest
