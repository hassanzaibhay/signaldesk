"""Shared Django settings.

Values come from the typed settings object in ``signaldesk.core.config``, never
from ``os.environ`` directly and never from literals in this file. ``DEBUG`` is
off here; only the dev module turns it on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from signaldesk.core.config import get_settings
from signaldesk.core.logging import django_logging_config

settings = get_settings()

# .../src/signaldesk/web/config/settings/base.py -> .../src/signaldesk
PACKAGE_DIR = Path(__file__).resolve().parents[3]
# .../src/signaldesk -> the repository root
BASE_DIR = PACKAGE_DIR.parents[1]

SECRET_KEY = settings.django_secret_key
DEBUG = False
ALLOWED_HOSTS = settings.allowed_hosts

INSTALLED_APPS = [
    "django_prometheus",
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "drf_spectacular",
    "django_htmx",
    "signaldesk.web.signals",
    "signaldesk.web.documents",
    "signaldesk.web.chat",
    "signaldesk.web.evalboard",
    "signaldesk.web.api",
]

MIDDLEWARE = [
    "django_prometheus.middleware.PrometheusBeforeMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "django_htmx.middleware.HtmxMiddleware",
    "django_prometheus.middleware.PrometheusAfterMiddleware",
]

ROOT_URLCONF = "signaldesk.web.config.urls"
WSGI_APPLICATION = "signaldesk.web.config.wsgi.application"
ASGI_APPLICATION = "signaldesk.web.config.asgi.application"

TEMPLATES: list[dict[str, Any]] = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [PACKAGE_DIR / "web" / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ]
        },
    }
]

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": settings.postgres_db,
        "USER": settings.postgres_user,
        "PASSWORD": settings.postgres_password,
        "HOST": settings.postgres_host,
        "PORT": str(settings.postgres_port),
        "CONN_MAX_AGE": 60,
        "OPTIONS": {"connect_timeout": 5},
    }
}

CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.redis.RedisCache",
        "LOCATION": settings.redis_url,
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATICFILES_DIRS = [PACKAGE_DIR / "web" / "static"]
STATIC_ROOT = settings.data_dir / "staticfiles"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

REST_FRAMEWORK = {
    "DEFAULT_SCHEMA_CLASS": "drf_spectacular.openapi.AutoSchema",
    "DEFAULT_PAGINATION_CLASS": "rest_framework.pagination.LimitOffsetPagination",
    "PAGE_SIZE": 50,
    "DEFAULT_RENDERER_CLASSES": ["rest_framework.renderers.JSONRenderer"],
}

SPECTACULAR_SETTINGS = {
    "TITLE": "SignalDesk API",
    "DESCRIPTION": (
        "Post-market drug safety signal intelligence. Disproportionality statistics "
        "are hypothesis-generating and do not establish causation."
    ),
    "VERSION": "0.1.0",
    "SERVE_INCLUDE_SCHEMA": False,
}

LOGGING = django_logging_config(debug=settings.django_debug)

#: Shown in every page footer and in every generated response. Not decoration:
#: the project fails its own evaluation if output implies causation.
HYPOTHESIS_NOTICE = (
    "Disproportionality statistics are hypothesis-generating. They quantify "
    "reporting patterns in a voluntary adverse event database and do not "
    "establish that a drug caused an event."
)
