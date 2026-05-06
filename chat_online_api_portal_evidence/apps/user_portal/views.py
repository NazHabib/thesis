from django.shortcuts import render

from apps.accounts.permissions import role_required
from apps.accounts.models import User
from apps.alerts.models import AlertEvent
from apps.profiles.models import PatientHealthRecord
from apps.recommendations.models import RecommendationRun


@role_required(User.Role.ADMIN, User.Role.CLINICIAN, User.Role.USER)
def dashboard(request):
    runs = (
        RecommendationRun.objects
        .filter(user=request.user)
        .prefetch_related("items")
        .order_by("-created_at")
    )

    latest_run = runs.first()
    latest_items = []
    if latest_run:
        latest_items = list(latest_run.items.order_by("priority", "id")[:3])

    open_alerts_count = AlertEvent.objects.filter(
        user=request.user,
        acknowledged=False,
    ).count()

    latest_health_record = (
        PatientHealthRecord.objects
        .filter(user=request.user)
        .order_by("-recorded_at", "-created_at")
        .first()
    )

    health_record_count = PatientHealthRecord.objects.filter(user=request.user).count()

    context = {
        "runs_count": runs.count(),
        "latest_run": latest_run,
        "latest_items": latest_items,
        "open_alerts_count": open_alerts_count,
        "latest_health_record": latest_health_record,
        "health_record_count": health_record_count,
    }
    return render(request, "user/dashboard.html", context)
