"""Routes owned by the signals app."""

from __future__ import annotations

from django.urls import URLPattern, path

from signaldesk.web.signals import views

app_name = "signals"

urlpatterns: list[URLPattern] = [
    path("", views.explorer, name="explorer"),
]
