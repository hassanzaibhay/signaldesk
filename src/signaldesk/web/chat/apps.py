"""App config for the bounded question-answering surface."""

from __future__ import annotations

from django.apps import AppConfig


class ChatConfig(AppConfig):
    name = "signaldesk.web.chat"
    label = "chat"
    verbose_name = "Ask"
    default_auto_field = "django.db.models.BigAutoField"
