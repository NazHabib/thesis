from django.http import HttpResponseForbidden
from django.shortcuts import get_object_or_404, render

from apps.accounts.permissions import role_required
from apps.accounts.models import User
from apps.profiles.models import PatientProfile
from apps.profiles.services import clinician_can_access_patient
from apps.recommendations.models import RecommendationRun

@role_required(User.Role.ADMIN, User.Role.CLINICIAN)
def patient_detail(request, patient_id: int):
    patient = get_object_or_404(User, id=patient_id)
    if not clinician_can_access_patient(request.user, patient):
        return HttpResponseForbidden("Forbidden: patient not assigned.")

    profile = PatientProfile.objects.filter(user=patient).first()
    rec_runs = RecommendationRun.objects.filter(user=patient).order_by("-created_at")[:10]

    return render(
        request,
        "clinician/patient_detail.html",
        {"patient": patient, "profile": profile, "rec_runs": rec_runs},
    )
