from django.conf import settings
from django.db import models


class AlertEvent(models.Model):
    class Severity(models.TextChoices):
        LOW = "LOW", "Low"
        MEDIUM = "MEDIUM", "Medium"
        HIGH = "HIGH", "High"
        CRITICAL = "CRITICAL", "Critical"

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="alerts",
    )

    severity = models.CharField(
        max_length=16,
        choices=Severity.choices,
        default=Severity.LOW,
    )

    title = models.CharField(max_length=200)
    message = models.TextField(blank=True, default="")

    # e.g., "bp_threshold", "glucose_trend", "manual_test"
    alert_type = models.CharField(max_length=50, default="manual_test")

    created_at = models.DateTimeField(auto_now_add=True)

    # workflow fields (optional but useful)
    acknowledged = models.BooleanField(default=False)
    acknowledged_at = models.DateTimeField(null=True, blank=True)

    escalated_to_clinician = models.BooleanField(default=False)

    def __str__(self):
        return f"[{self.severity}] {self.title} ({self.user})"

