from django.shortcuts import render
from apps.accounts.permissions import role_required
from apps.accounts.models import User
from apps.alerts.models import AlertEvent
from apps.profiles.models import ClinicianPatientLink


@role_required(User.Role.ADMIN, User.Role.CLINICIAN)
def alerts_list(request):
    # Admin sees all alerts; clinician sees alerts for assigned patients
    if request.user.role == User.Role.ADMIN:
        alerts = AlertEvent.objects.select_related("user").order_by("-created_at")[:200]
    else:
        patient_ids = ClinicianPatientLink.objects.filter(
            clinician=request.user
        ).values_list("patient_id", flat=True)

        alerts = AlertEvent.objects.filter(
            user_id__in=patient_ids
        ).select_related("user").order_by("-created_at")[:200]

    return render(request, "clinician/alerts.html", {"alerts": alerts})
