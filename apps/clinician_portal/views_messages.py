from django.contrib import messages
from django.http import HttpResponseForbidden
from django.shortcuts import get_object_or_404, redirect, render

from apps.accounts.models import User
from apps.accounts.permissions import role_required
from apps.alerts.models import AlertEvent
from apps.profiles.services import clinician_can_access_patient
from apps.recommendations.models import RecommendationRun


@role_required(User.Role.ADMIN, User.Role.CLINICIAN)
def send_run_message(request, patient_id: int, run_id: int):
    patient = get_object_or_404(User, id=patient_id)
    run = get_object_or_404(RecommendationRun, id=run_id, user=patient)

    if not clinician_can_access_patient(request.user, patient):
        return HttpResponseForbidden("Forbidden: patient not assigned.")

    if request.method == "POST":
        title = request.POST.get("title", "").strip()
        message = request.POST.get("message", "").strip()
        severity = request.POST.get("severity", AlertEvent.Severity.LOW).strip()
        clinician_note = request.POST.get("clinician_note", "").strip()

        form = {
            "title": title,
            "message": message,
            "severity": severity,
            "clinician_note": clinician_note,
        }

        if not title:
            messages.error(request, "Please enter a message title.")
            return render(
                request,
                "clinician/send_run_message.html",
                {
                    "patient": patient,
                    "run": run,
                    "severity_choices": AlertEvent.Severity.choices,
                    "form": form,
                },
            )

        if not message:
            messages.error(request, "Please enter a patient message.")
            return render(
                request,
                "clinician/send_run_message.html",
                {
                    "patient": patient,
                    "run": run,
                    "severity_choices": AlertEvent.Severity.choices,
                    "form": form,
                },
            )

        AlertEvent.objects.create(
            user=patient,
            created_by=request.user,
            recommendation_run=run,
            severity=severity,
            visibility=AlertEvent.Visibility.PATIENT_VISIBLE,
            title=title,
            message=message,
            alert_type="clinician_message",
            clinician_note=clinician_note,
            escalated_to_clinician=True,
        )

        messages.success(request, f"Message sent to {patient.username}.")
        return redirect(
            "clinician:patient_run_detail",
            patient_id=patient.id,
            run_id=run.id,
        )

    default_title = f"Clinician review for recommendation run #{run.id}"
    default_message = (
        "I reviewed your latest recommendation summary. Please focus first on the highest-priority "
        "recommendations and bring any recent health measurements or questions to your next care discussion."
    )

    return render(
        request,
        "clinician/send_run_message.html",
        {
            "patient": patient,
            "run": run,
            "severity_choices": AlertEvent.Severity.choices,
            "form": {
                "title": default_title,
                "message": default_message,
                "severity": AlertEvent.Severity.LOW,
                "clinician_note": "",
            },
        },
    )


@role_required(User.Role.ADMIN, User.Role.CLINICIAN)
def send_patient_message(request, patient_id: int):
    patient = get_object_or_404(User, id=patient_id)

    if not clinician_can_access_patient(request.user, patient):
        return HttpResponseForbidden("Forbidden: patient not assigned.")

    if request.method == "POST":
        title = request.POST.get("title", "").strip()
        message = request.POST.get("message", "").strip()
        severity = request.POST.get("severity", AlertEvent.Severity.LOW).strip()
        clinician_note = request.POST.get("clinician_note", "").strip()

        form = {
            "title": title,
            "message": message,
            "severity": severity,
            "clinician_note": clinician_note,
        }

        if not title:
            messages.error(request, "Please enter a message title.")
            return render(
                request,
                "clinician/send_patient_message.html",
                {
                    "patient": patient,
                    "severity_choices": AlertEvent.Severity.choices,
                    "form": form,
                },
            )

        if not message:
            messages.error(request, "Please enter a patient message.")
            return render(
                request,
                "clinician/send_patient_message.html",
                {
                    "patient": patient,
                    "severity_choices": AlertEvent.Severity.choices,
                    "form": form,
                },
            )

        AlertEvent.objects.create(
            user=patient,
            created_by=request.user,
            recommendation_run=None,
            severity=severity,
            visibility=AlertEvent.Visibility.PATIENT_VISIBLE,
            title=title,
            message=message,
            alert_type="clinician_message",
            clinician_note=clinician_note,
            escalated_to_clinician=True,
        )

        messages.success(request, f"Message sent to {patient.username}.")
        return redirect("clinician:patient_detail", patient_id=patient.id)

    return render(
        request,
        "clinician/send_patient_message.html",
        {
            "patient": patient,
            "severity_choices": AlertEvent.Severity.choices,
            "form": {
                "title": "",
                "message": "",
                "severity": AlertEvent.Severity.LOW,
                "clinician_note": "",
            },
        },
    )
