from django.shortcuts import render
from apps.accounts.permissions import role_required
from apps.accounts.models import User
from apps.profiles.models import ClinicianPatientLink


@role_required(User.Role.ADMIN, User.Role.CLINICIAN)
def patients(request):
    links = ClinicianPatientLink.objects.filter(clinician=request.user).select_related("patient")
    return render(request, "clinician/patients.html", {"links": links})
