from django.contrib import admin

from .models import AlertEvent


@admin.register(AlertEvent)
class AlertEventAdmin(admin.ModelAdmin):
    list_display = (
        "created_at",
        "user",
        "created_by",
        "severity",
        "visibility",
        "title",
        "alert_type",
        "acknowledged",
        "escalated_to_clinician",
        "recommendation_run",
    )

    list_filter = (
        "severity",
        "visibility",
        "alert_type",
        "acknowledged",
        "escalated_to_clinician",
        "created_at",
    )

    search_fields = (
        "user__username",
        "user__email",
        "created_by__username",
        "title",
        "message",
        "clinician_note",
    )

    readonly_fields = ("created_at",)

    ordering = ("-created_at",)
