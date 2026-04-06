from django.shortcuts import render
from apps.accounts.permissions import role_required
from apps.accounts.models import User
from apps.recommendations.models import RecommendationRun
# from apps.alerts.models import AlertEvent

@role_required(User.Role.ADMIN, User.Role.CLINICIAN, User.Role.USER)
def dashboard(request):
    runs_count = RecommendationRun.objects.filter(user=request.user).count()
    return render(request, "user/dashboard.html", {"runs_count": runs_count})

