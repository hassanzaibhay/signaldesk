"""URL configuration.

Only the routes that exist today: the health endpoint, the OpenAPI schema, the
admin, and the Prometheus metrics endpoint. The application routes arrive with
the apps that own them.
"""

from __future__ import annotations

from django.contrib import admin
from django.urls import URLPattern, URLResolver, include, path
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView

from signaldesk.web.health import healthz

urlpatterns: list[URLPattern | URLResolver] = [
    path("healthz/", healthz, name="healthz"),
    path("api/schema/", SpectacularAPIView.as_view(), name="schema"),
    path(
        "api/schema/swagger-ui/",
        SpectacularSwaggerView.as_view(url_name="schema"),
        name="swagger-ui",
    ),
    path("admin/", admin.site.urls),
    path("", include("django_prometheus.urls")),
]
