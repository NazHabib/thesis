from datetime import date

from django.contrib import messages
from django.contrib.auth import logout
from django.shortcuts import redirect, render

from apps.accounts.permissions import role_required
from apps.accounts.models import User
from apps.profiles.models import PatientProfile


def _float_or_none(value: str):
    value = (value or "").strip()
    if not value:
        return None
    return float(value)


def _int_or_none(value: str):
    value = (value or "").strip()
    if not value:
        return None
    return int(value)


def _date_or_none(value: str):
    value = (value or "").strip()
    if not value:
        return None
    return date.fromisoformat(value)


@role_required(User.Role.ADMIN, User.Role.CLINICIAN, User.Role.USER)
def profile(request):
    prof, _ = PatientProfile.objects.get_or_create(user=request.user)
    return render(request, "user/profile.html", {"profile": prof})


@role_required(User.Role.ADMIN, User.Role.CLINICIAN, User.Role.USER)
def account_edit(request):
    if request.method == "POST":
        request.user.first_name = request.POST.get("first_name", "").strip()
        request.user.last_name = request.POST.get("last_name", "").strip()
        request.user.email = request.POST.get("email", "").strip()
        request.user.save(update_fields=["first_name", "last_name", "email"])

        messages.success(request, "Account information updated.")
        return redirect("user:profile")

    return render(request, "user/account_edit.html")


@role_required(User.Role.ADMIN, User.Role.CLINICIAN, User.Role.USER)
def health_edit(request):
    prof, _ = PatientProfile.objects.get_or_create(user=request.user)

    if request.method == "POST":
        try:
            prof.has_dm = bool(request.POST.get("has_dm"))
            prof.has_htn = bool(request.POST.get("has_htn"))

            prof.sex = request.POST.get("sex", "").strip()
            prof.date_of_birth = _date_or_none(request.POST.get("date_of_birth", ""))

            prof.target_glucose_min = _float_or_none(request.POST.get("target_glucose_min", ""))
            prof.target_glucose_max = _float_or_none(request.POST.get("target_glucose_max", ""))
            prof.target_bp_sys = _int_or_none(request.POST.get("target_bp_sys", ""))
            prof.target_bp_dia = _int_or_none(request.POST.get("target_bp_dia", ""))

            prof.save()
            messages.success(request, "Health information updated.")
            return redirect("user:profile")

        except ValueError:
            messages.error(request, "Please check the numbers and date entered in your health information.")

    return render(request, "user/health_edit.html", {"profile": prof})


@role_required(User.Role.USER)
def account_delete(request):
    if request.method == "POST":
        confirm_text = request.POST.get("confirm_text", "").strip()

        if confirm_text != "DELETE":
            messages.error(request, "Please type DELETE to confirm account deletion.")
            return redirect("user:account_delete")

        user = request.user
        logout(request)
        user.delete()

        messages.success(request, "Your account has been deleted.")
        return redirect("login")

    return render(request, "user/account_delete.html")
