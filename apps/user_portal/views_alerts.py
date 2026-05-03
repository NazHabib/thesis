from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone

from apps.accounts.models import User
from apps.accounts.permissions import role_required
from apps.alerts.models import AlertEvent


@role_required(User.Role.ADMIN, User.Role.CLINICIAN, User.Role.USER)
def alerts_list(request):
    alerts = (
        AlertEvent.objects
        .filter(user=request.user, visibility=AlertEvent.Visibility.PATIENT_VISIBLE)
        .order_by("-created_at")
    )

    open_count = alerts.filter(acknowledged=False).count()
    reviewed_count = alerts.filter(acknowledged=True).count()

    return render(
        request,
        "user/alerts.html",
        {
            "alerts": alerts,
            "open_count": open_count,
            "reviewed_count": reviewed_count,
        },
    )


@role_required(User.Role.ADMIN, User.Role.CLINICIAN, User.Role.USER)
def alert_ack(request, alert_id: int):
    alert = get_object_or_404(
        AlertEvent,
        id=alert_id,
        user=request.user,
        visibility=AlertEvent.Visibility.PATIENT_VISIBLE,
    )

    alert.acknowledged = True
    alert.acknowledged_at = timezone.now()
    alert.save(update_fields=["acknowledged", "acknowledged_at"])

    return redirect("user:alerts")
