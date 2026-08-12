"""WSGI entrypoint, used by gunicorn in the production image."""

from __future__ import annotations

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "signaldesk.web.config.settings.dev")

application = get_wsgi_application()
