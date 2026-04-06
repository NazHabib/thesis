from django.contrib import messages
from django.shortcuts import redirect, render

from apps.accounts.permissions import role_required
from apps.accounts.models import User
from apps.profiles.models import PatientProfile


@role_required(User.Role.ADMIN, User.Role.CLINICIAN, User.Role.USER)
def profile(request):
    prof, _ = PatientProfile.objects.get_or_create(user=request.user)

    if request.method == "POST":
        prof.has_dm = bool(request.POST.get("has_dm"))
        prof.has_htn = bool(request.POST.get("has_htn"))

        prof.sex = request.POST.get("sex", "").strip()
        dob = request.POST.get("date_of_birth", "").strip()
        prof.date_of_birth = dob or None

        def f(name):
            v = request.POST.get(name, "").strip()
            return float(v) if v else None

        def i(name):
            v = request.POST.get(name, "").strip()
            return int(v) if v else None

        prof.target_glucose_min = f("target_glucose_min")
        prof.target_glucose_max = f("target_glucose_max")
        prof.target_bp_sys = i("target_bp_sys")
        prof.target_bp_dia = i("target_bp_dia")

        prof.save()
        messages.success(request, "Profile updated.")
        return redirect("user:profile")

    return render(request, "user/profile.html", {"profile": prof})
