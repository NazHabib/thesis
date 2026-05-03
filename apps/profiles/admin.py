from django.contrib import admin

from .models import PatientProfile, ClinicianPatientLink, PatientHealthRecord


@admin.register(PatientProfile)
class PatientProfileAdmin(admin.ModelAdmin):
    list_display = ("user", "has_dm", "has_htn", "updated_at")
    search_fields = ("user__username", "user__email")
    list_filter = ("has_dm", "has_htn")


@admin.register(PatientHealthRecord)
class PatientHealthRecordAdmin(admin.ModelAdmin):
    list_display = (
        "user",
        "recorded_at",
        "source",
        "bmi",
        "hba1c",
        "systolic_bp",
        "diastolic_bp",
        "updated_at",
    )
    search_fields = ("user__username", "user__email", "label", "notes")
    list_filter = (
        "source",
        "current_smoker",
        "alcohol_over_guideline",
        "known_prediabetes",
        "known_severe_hyperglycemia",
        "known_hypertensive_crisis",
        "recorded_at",
    )
    readonly_fields = ("created_at", "updated_at")


@admin.register(ClinicianPatientLink)
class ClinicianPatientLinkAdmin(admin.ModelAdmin):
    list_display = ("clinician", "patient", "created_at")
    search_fields = (
        "clinician__username",
        "clinician__email",
        "patient__username",
        "patient__email",
    )
    list_filter = ("clinician", "created_at")
    autocomplete_fields = ("clinician", "patient")
