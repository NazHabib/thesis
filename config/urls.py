"""
URL configuration for config project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from apps.accounts.views import home_redirect
from django.contrib.auth import views as auth_views
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path("admin/", admin.site.urls),
    path("", home_redirect, name="home"),
    path("user/", include("apps.user_portal.urls")),
    path("clinician/", include("apps.clinician_portal.urls")),
    path("testing/", include("apps.testing_portal.urls")),
    path("login/", auth_views.LoginView.as_view(template_name="registration/login.html"), name="login"),
    path("logout/", auth_views.LogoutView.as_view(), name="logout"),
    path("recs/", include("apps.recommendations.urls")),
    path("de/", include("apps.pghd.urls_de")),
    path("api/recs-csl/", include("apps.recs_csl.urls")),
    path("api/ranking-layer/", include("apps.ranking_layer.urls")),
    path("api/bundling-layer/", include("apps.bundling_layer.urls")),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
