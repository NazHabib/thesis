from django.conf import settings
from django.db import models


class PatientProfile(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="patient_profile")

    # Minimal clinical context (expand later)
    date_of_birth = models.DateField(null=True, blank=True)
    sex = models.CharField(max_length=20, blank=True, default="")  # optional

    has_dm = models.BooleanField(default=False)
    has_htn = models.BooleanField(default=False)

    # Targets (simple placeholders)
    target_glucose_min = models.FloatField(null=True, blank=True)
    target_glucose_max = models.FloatField(null=True, blank=True)
    target_bp_sys = models.IntegerField(null=True, blank=True)
    target_bp_dia = models.IntegerField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)


class ClinicianPatientLink(models.Model):
    clinician = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="assigned_patients"
    )
    patient = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="assigned_clinicians"
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("clinician", "patient")

    def __str__(self):
        return f"{self.clinician.username} -> {self.patient.username}"

