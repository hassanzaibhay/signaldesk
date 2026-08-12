"""App config for the signal explorer."""

from __future__ import annotations

from django.apps import AppConfig


class SignalsConfig(AppConfig):
    name = "signaldesk.web.signals"
    label = "signals"
    verbose_name = "Signals"
    default_auto_field = "django.db.models.BigAutoField"
