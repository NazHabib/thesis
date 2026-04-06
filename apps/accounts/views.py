from django.shortcuts import redirect
from django.contrib.auth.decorators import login_required
from .models import User

@login_required
def home_redirect(request):
    role = request.user.role
    if role == User.Role.ADMIN:
        return redirect("/admin/")
    if role == User.Role.CLINICIAN:
        return redirect("clinician:dashboard")
    if role == User.Role.TESTER:
        return redirect("testing:dashboard")
    return redirect("user:dashboard")

