"""App config for source documents: labels, trials, literature."""

from __future__ import annotations

from django.apps import AppConfig


class DocumentsConfig(AppConfig):
    name = "signaldesk.web.documents"
    label = "documents"
    verbose_name = "Documents"
    default_auto_field = "django.db.models.BigAutoField"
