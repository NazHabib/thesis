from django.contrib import admin
from .models import AlertEvent


@admin.register(AlertEvent)
class AlertEventAdmin(admin.ModelAdmin):
    list_display = ("created_at", "user", "severity", "title", "acknowledged", "escalated_to_clinician")
    list_filter = ("severity", "acknowledged", "escalated_to_clinician")
    search_fields = ("user__username", "title", "message")
    ordering = ("-created_at",)

