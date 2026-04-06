from django.http import HttpResponseForbidden
from django.shortcuts import get_object_or_404, render

from apps.accounts.permissions import role_required
from apps.accounts.models import User
from apps.profiles.services import clinician_can_access_patient
from apps.recommendations.models import RecommendationRun

@role_required(User.Role.ADMIN, User.Role.CLINICIAN)
def patient_run_detail(request, patient_id: int, run_id: int):
    patient = get_object_or_404(User, id=patient_id)
    if not clinician_can_access_patient(request.user, patient):
        return HttpResponseForbidden("Forbidden: patient not assigned.")

    run = get_object_or_404(RecommendationRun, id=run_id, user=patient)
    return render(request, "clinician/patient_run_detail.html", {"patient": patient, "run": run})
