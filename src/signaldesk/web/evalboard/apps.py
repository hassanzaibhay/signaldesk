"""App config for the public evaluation dashboard."""

from __future__ import annotations

from django.apps import AppConfig


class EvalboardConfig(AppConfig):
    name = "signaldesk.web.evalboard"
    label = "evalboard"
    verbose_name = "Evaluation"
    default_auto_field = "django.db.models.BigAutoField"
