from django.contrib import admin
from .models import PatientProfile, ClinicianPatientLink


@admin.register(PatientProfile)
class PatientProfileAdmin(admin.ModelAdmin):
    list_display = ("user", "has_dm", "has_htn", "updated_at")
    search_fields = ("user__username", "user__email")
    list_filter = ("has_dm", "has_htn")


@admin.register(ClinicianPatientLink)
class ClinicianPatientLinkAdmin(admin.ModelAdmin):
    list_display = ("clinician", "patient", "created_at")
    search_fields = ("clinician__username", "patient__username")

