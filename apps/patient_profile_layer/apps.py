from __future__ import annotations

from django.apps import AppConfig


class PatientProfileLayerConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.patient_profile_layer"
    label = "patient_profile_layer"
    verbose_name = "Patient Profile Layer"
