from django.http import HttpResponseForbidden
from django.shortcuts import get_object_or_404, render

from apps.accounts.permissions import role_required
from apps.accounts.models import User
from apps.profiles.models import PatientProfile, PatientHealthRecord
from apps.profiles.services import clinician_can_access_patient, build_current_health_feature_payload
from apps.recommendations.models import RecommendationRun


@role_required(User.Role.ADMIN, User.Role.CLINICIAN)
def patient_detail(request, patient_id: int):
    patient = get_object_or_404(User, id=patient_id)

    if not clinician_can_access_patient(request.user, patient):
        return HttpResponseForbidden("Forbidden: patient not assigned.")

    profile = PatientProfile.objects.filter(user=patient).first()

    health_records = (
        PatientHealthRecord.objects
        .filter(user=patient)
        .order_by("-recorded_at", "-created_at")
    )

    latest_health_record = health_records.first()
    recent_health_records = health_records[:5]

    current_health_payload = build_current_health_feature_payload(patient)

    rec_runs = (
        RecommendationRun.objects
        .filter(user=patient)
        .prefetch_related("items")
        .order_by("-created_at")[:10]
    )

    return render(
        request,
        "clinician/patient_detail.html",
        {
            "patient": patient,
            "profile": profile,
            "latest_health_record": latest_health_record,
            "recent_health_records": recent_health_records,
            "health_record_count": health_records.count(),
            "current_health_payload": current_health_payload,
            "rec_runs": rec_runs,
        },
    )
