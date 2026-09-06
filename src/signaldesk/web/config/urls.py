"""URL configuration.

Only the routes that exist today: the signals explorer, the health endpoint,
the OpenAPI schema, the admin, and the Prometheus metrics endpoint. The
remaining application routes arrive with the apps that own them.

The explorer owns ``/signals/``. The root redirects to it rather than serving
it: one URL for one page keeps a shared link unambiguous, and a root that 404s
on a running application reads as a broken deployment.
"""

from __future__ import annotations

from django.contrib import admin
from django.urls import URLPattern, URLResolver, include, path
from django.views.generic import RedirectView
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView

from signaldesk.web.health import healthz

urlpatterns: list[URLPattern | URLResolver] = [
    path("", RedirectView.as_view(pattern_name="signals:explorer"), name="root"),
    path("signals/", include("signaldesk.web.signals.urls")),
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
