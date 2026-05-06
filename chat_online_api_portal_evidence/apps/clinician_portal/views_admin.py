from django.contrib import messages
from django.db.models import Count, Q
from django.shortcuts import get_object_or_404, redirect, render

from apps.accounts.models import User
from apps.accounts.permissions import role_required
from apps.profiles.models import ClinicianPatientLink
from apps.recommendations.models import RecommendationRun


@role_required(User.Role.ADMIN)
def admin_tools(request):
    users = User.objects.all()

    patient_count = users.filter(role=User.Role.USER).count()
    clinician_count = users.filter(role=User.Role.CLINICIAN).count()
    admin_count = users.filter(role=User.Role.ADMIN).count()
    assignment_count = ClinicianPatientLink.objects.count()

    runs = RecommendationRun.objects.all()
    live_runs_count = runs.filter(engine_version="live_single_patient_pipeline").count()
    fallback_runs_count = runs.filter(engine_version="reference_match_fallback").count()
    total_runs_count = runs.count()

    latest_runs = (
        runs.select_related("user")
        .annotate(item_count=Count("items"))
        .order_by("-created_at")[:5]
    )

    return render(
        request,
        "clinician/admin_tools.html",
        {
            "patient_count": patient_count,
            "clinician_count": clinician_count,
            "admin_count": admin_count,
            "assignment_count": assignment_count,
            "total_runs_count": total_runs_count,
            "live_runs_count": live_runs_count,
            "fallback_runs_count": fallback_runs_count,
            "latest_runs": latest_runs,
        },
    )


@role_required(User.Role.ADMIN)
def patient_assignments(request):
    clinicians = User.objects.filter(role=User.Role.CLINICIAN).order_by("username")
    patients = User.objects.filter(role=User.Role.USER).order_by("username")

    if request.method == "POST":
        clinician_id = request.POST.get("clinician_id")
        patient_id = request.POST.get("patient_id")

        clinician = get_object_or_404(User, id=clinician_id, role=User.Role.CLINICIAN)
        patient = get_object_or_404(User, id=patient_id, role=User.Role.USER)

        link, created = ClinicianPatientLink.objects.get_or_create(
            clinician=clinician,
            patient=patient,
        )

        if created:
            messages.success(
                request,
                f"Assigned patient {patient.username} to clinician {clinician.username}.",
            )
        else:
            messages.info(
                request,
                f"{patient.username} is already assigned to {clinician.username}.",
            )

        return redirect("clinician:patient_assignments")

    assignments = (
        ClinicianPatientLink.objects
        .select_related("clinician", "patient")
        .order_by("clinician__username", "patient__username")
    )

    return render(
        request,
        "clinician/patient_assignments.html",
        {
            "clinicians": clinicians,
            "patients": patients,
            "assignments": assignments,
        },
    )


@role_required(User.Role.ADMIN)
def remove_patient_assignment(request, link_id: int):
    link = get_object_or_404(
        ClinicianPatientLink.objects.select_related("clinician", "patient"),
        id=link_id,
    )

    if request.method != "POST":
        messages.warning(request, "Use the remove button to delete an assignment.")
        return redirect("clinician:patient_assignments")

    clinician_name = link.clinician.username
    patient_name = link.patient.username
    link.delete()

    messages.success(
        request,
        f"Removed assignment: {patient_name} from {clinician_name}.",
    )
    return redirect("clinician:patient_assignments")


@role_required(User.Role.ADMIN)
def pipeline_runs(request):
    mode = request.GET.get("mode", "").strip()

    runs = (
        RecommendationRun.objects
        .select_related("user")
        .annotate(item_count=Count("items"))
        .order_by("-created_at")
    )

    if mode == "live":
        runs = runs.filter(engine_version="live_single_patient_pipeline")
    elif mode == "fallback":
        runs = runs.filter(engine_version="reference_match_fallback")
    elif mode == "old":
        runs = runs.exclude(
            engine_version__in=[
                "live_single_patient_pipeline",
                "reference_match_fallback",
            ]
        )

    return render(
        request,
        "clinician/pipeline_runs.html",
        {
            "runs": runs[:100],
            "mode": mode,
            "live_count": RecommendationRun.objects.filter(
                engine_version="live_single_patient_pipeline"
            ).count(),
            "fallback_count": RecommendationRun.objects.filter(
                engine_version="reference_match_fallback"
            ).count(),
            "old_count": RecommendationRun.objects.exclude(
                engine_version__in=[
                    "live_single_patient_pipeline",
                    "reference_match_fallback",
                ]
            ).count(),
        },
    )
