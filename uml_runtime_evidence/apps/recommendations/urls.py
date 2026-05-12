from django.urls import path
from . import views

app_name = "recs"

urlpatterns = [
    path("generate/", views.generate, name="generate"),
    path("history/", views.history, name="history"),
    path("<int:run_id>/", views.detail, name="detail"),
]
