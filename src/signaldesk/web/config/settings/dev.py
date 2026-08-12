"""Development settings: local stack, verbose errors, no security theatre."""

from __future__ import annotations

from signaldesk.core.config import get_settings

from .base import *  # noqa: F403

_settings = get_settings()

DEBUG = _settings.django_debug
ALLOWED_HOSTS = [*_settings.allowed_hosts, "web", "localhost", "127.0.0.1"]
INTERNAL_IPS = ["127.0.0.1"]
