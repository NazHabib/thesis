from django.conf import settings
from django.db import models


class AlertEvent(models.Model):
    class Severity(models.TextChoices):
        LOW = "LOW", "Low"
        MEDIUM = "MEDIUM", "Medium"
        HIGH = "HIGH", "High"
        CRITICAL = "CRITICAL", "Critical"

    class Visibility(models.TextChoices):
        PATIENT_VISIBLE = "patient_visible", "Patient visible"
        CLINICIAN_ONLY = "clinician_only", "Clinician only"

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="alerts",
    )

    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="created_alerts",
    )

    recommendation_run = models.ForeignKey(
        "recommendations.RecommendationRun",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="clinician_messages",
    )

    severity = models.CharField(
        max_length=16,
        choices=Severity.choices,
        default=Severity.LOW,
    )

    visibility = models.CharField(
        max_length=32,
        choices=Visibility.choices,
        default=Visibility.PATIENT_VISIBLE,
    )

    title = models.CharField(max_length=200)
    message = models.TextField(blank=True, default="")

    # e.g. bp_threshold, glucose_trend, manual_test, clinician_message
    alert_type = models.CharField(max_length=50, default="manual_test")

    clinician_note = models.TextField(blank=True, default="")

    created_at = models.DateTimeField(auto_now_add=True)

    acknowledged = models.BooleanField(default=False)
    acknowledged_at = models.DateTimeField(null=True, blank=True)

    escalated_to_clinician = models.BooleanField(default=False)

    def __str__(self):
        return f"[{self.severity}] {self.title} ({self.user})"
