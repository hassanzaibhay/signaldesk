"""Test settings.

Unit tests must run on a bare checkout with no containers and no ``.env`` file,
including on Windows, so this module supplies placeholder values for anything
that would otherwise be required. Tests that need a real database are marked
``integration`` and run against the compose stack, which supplies the real
environment.
"""

from __future__ import annotations

import os

os.environ.setdefault("DJANGO_SECRET_KEY", "test-only-not-a-secret")
os.environ.setdefault("DJANGO_DEBUG", "0")
os.environ.setdefault("DATA_DIR", os.environ.get("TMPDIR", "."))
os.environ.setdefault("CACHE_DIR", os.environ.get("TMPDIR", "."))

from .base import *  # noqa: F403

DEBUG = False
ALLOWED_HOSTS = ["testserver", "localhost", "127.0.0.1"]

CACHES = {"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"}}

PASSWORD_HASHERS = ["django.contrib.auth.hashers.MD5PasswordHasher"]

# Keep test output readable; the JSON handler is exercised by its own unit test.
LOGGING = {"version": 1, "disable_existing_loggers": False, "root": {"level": "CRITICAL"}}
