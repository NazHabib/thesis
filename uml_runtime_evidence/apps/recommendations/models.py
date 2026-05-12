from django.conf import settings
from django.db import models


class RecommendationRun(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    engine_version = models.CharField(max_length=64, default="live_single_patient_pipeline")
    notes = models.TextField(blank=True, default="")

    clinician_reviewed = models.BooleanField(default=False)
    clinician_reviewed_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="reviewed_recommendation_runs",
    )
    clinician_reviewed_at = models.DateTimeField(null=True, blank=True)
    clinician_review_note = models.TextField(blank=True, default="")

    def __str__(self):
        return f"Run #{self.id} for {self.user} at {self.created_at:%Y-%m-%d %H:%M}"


class RecommendationItem(models.Model):
    run = models.ForeignKey(RecommendationRun, related_name="items", on_delete=models.CASCADE)
    title = models.CharField(max_length=200)
    category = models.CharField(max_length=50, default="lifestyle")  # lifestyle/medication/monitoring/etc
    priority = models.IntegerField(default=3)  # 1 high, 5 low
    rationale = models.TextField(blank=True, default="")
    explanation = models.TextField(blank=True, default="")  # XAI text placeholder
    source = models.CharField(max_length=32, default="live_pipeline")
    confidence = models.FloatField(default=0.0)

    def __str__(self):
        return f"{self.title} (p{self.priority})"
