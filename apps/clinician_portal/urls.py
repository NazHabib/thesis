from django.urls import path

from . import views
from . import views_patients
from . import views_patient_detail
from . import views_patient_run
from . import views_alerts
from . import views_admin
from . import views_messages
from . import views_patient_health

app_name = "clinician"

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("admin-tools/", views_admin.admin_tools, name="admin_tools"),
    path("admin-tools/assignments/", views_admin.patient_assignments, name="patient_assignments"),
    path("admin-tools/assignments/<int:link_id>/remove/", views_admin.remove_patient_assignment, name="remove_patient_assignment"),
    path("admin-tools/runs/", views_admin.pipeline_runs, name="pipeline_runs"),
    path("patients/", views_patients.patients, name="patients"),
    path("patient/<int:patient_id>/", views_patient_detail.patient_detail, name="patient_detail"),
    path("patients/<int:patient_id>/runs/<int:run_id>/", views_patient_run.patient_run_detail, name="patient_run_detail"),
    path("patients/<int:patient_id>/runs/<int:run_id>/review/",views_patient_run.patient_run_review, name="patient_run_review"),
    path("patients/<int:patient_id>/runs/<int:run_id>/message/", views_messages.send_run_message, name="send_run_message"),
    path("patient/<int:patient_id>/message/", views_messages.send_patient_message,name="send_patient_message"),
    path("thesis-demo/", views_patient_run.thesis_demo_runs, name="thesis_demo"),
    path("thesis-demo/<str:participant_key>/", views_patient_run.thesis_demo_patient, name="thesis_demo_patient"),
    path("system-evidence/", views.system_evidence, name="system_evidence"),
    path("model-registry/", views.model_registry, name="model_registry"),
    path("dataset-features/", views.dataset_features, name="dataset_features"),
    path("system-behaviour/", views.system_behaviour, name="system_behaviour"),
    path("model-performance/", views.model_performance, name="model_performance"),
    path("alerts/", views_alerts.alerts_list, name="alerts"),
    path(
        "patient/<int:patient_id>/health-data/add/",
        views_patient_health.clinician_health_data_add,
        name="clinician_health_data_add",
    ),
    path(
        "patient/<int:patient_id>/generate-recommendations/",
        views_patient_health.clinician_generate_patient_recommendations,
        name="clinician_generate_patient_recommendations",
    ),
]
