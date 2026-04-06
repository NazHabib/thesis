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

    path("alerts/", views_alerts.alerts_list, name="alerts"),
]
