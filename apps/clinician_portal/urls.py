from django.urls import path

from . import views
from . import views_patients
from . import views_patient_detail
from . import views_patient_run
from . import views_alerts

app_name = "clinician"

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("patients/", views_patients.patients, name="patients"),
    path("patient/<int:patient_id>/", views_patient_detail.patient_detail, name="patient_detail"),
    path("patients/<int:patient_id>/runs/<int:run_id>/", views_patient_run.patient_run_detail, name="patient_run_detail"),
    path("thesis-demo/", views_patient_run.thesis_demo_runs, name="thesis_demo"),
    path("thesis-demo/<str:participant_key>/", views_patient_run.thesis_demo_patient, name="thesis_demo_patient"),
    path("system-evidence/", views.system_evidence, name="system_evidence"),
    path("model-registry/", views.model_registry, name="model_registry"),
    path("dataset-features/", views.dataset_features, name="dataset_features"),
    path("system-behaviour/", views.system_behaviour, name="system_behaviour"),
    path("model-performance/", views.model_performance, name="model_performance"),
    path("alerts/", views_alerts.alerts_list, name="alerts"),
]
