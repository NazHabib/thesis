from django.contrib import messages
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone

from apps.accounts.permissions import role_required
from apps.accounts.models import User
from apps.alerts.models import AlertEvent


@role_required(User.Role.ADMIN, User.Role.CLINICIAN, User.Role.USER)
def alerts_list(request):
    alerts = AlertEvent.objects.filter(user=request.user).order_by("-created_at")
    return render(request, "user/alerts.html", {"alerts": alerts})


@role_required(User.Role.ADMIN, User.Role.CLINICIAN, User.Role.USER)
def alert_ack(request, alert_id: int):
    alert = get_object_or_404(AlertEvent, id=alert_id, user=request.user)

    alert.acknowledged = True
    alert.acknowledged_at = timezone.now()
    alert.save(update_fields=["acknowledged", "acknowledged_at"])

    messages.success(request, "Alert acknowledged.")
    return redirect("user:alerts")
