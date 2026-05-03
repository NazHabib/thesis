from django.urls import path
from . import views_de

app_name = "de"

urlpatterns = [
    path("datasets/", views_de.datasets, name="datasets"),
    path("datasets/new/", views_de.dataset_create, name="dataset_create"),
    path("datasets/<int:dataset_id>/", views_de.dataset_detail, name="dataset_detail"),
    path("datasets/<int:dataset_id>/upload/", views_de.dataset_upload, name="dataset_upload"),
    path("datasets/<int:dataset_id>/run-profile/", views_de.run_profile, name="run_profile"),
]
