from django.urls import path
from .views import BundlingRunView

urlpatterns = [
    path("run/", BundlingRunView.as_view(), name="bundling-run"),
]
