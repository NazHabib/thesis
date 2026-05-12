from django.shortcuts import render

from apps.accounts.permissions import role_required
from apps.accounts.models import User
from apps.profiles.models import ClinicianPatientLink


@role_required(User.Role.ADMIN, User.Role.CLINICIAN)
def patients(request):
    if request.user.role == User.Role.ADMIN:
        links = (
            ClinicianPatientLink.objects
            .select_related("clinician", "patient")
            .order_by("clinician__username", "patient__username")
        )
    else:
        links = (
            ClinicianPatientLink.objects
            .filter(clinician=request.user)
            .select_related("patient")
            .order_by("patient__username")
        )

    return render(request, "clinician/patients.html", {"links": links})
