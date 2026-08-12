"""App config for the REST API."""

from __future__ import annotations

from django.apps import AppConfig


class ApiConfig(AppConfig):
    name = "signaldesk.web.api"
    label = "api"
    verbose_name = "API"
    default_auto_field = "django.db.models.BigAutoField"
