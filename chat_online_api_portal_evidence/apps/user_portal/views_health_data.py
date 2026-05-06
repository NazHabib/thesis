from __future__ import annotations

from django.contrib import messages
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone

from apps.accounts.models import User
from apps.accounts.permissions import role_required
from apps.profiles.models import PatientHealthRecord
from apps.profiles.services import build_current_health_feature_payload
from django.utils.dateparse import parse_date


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


def _validate_range(errors: list[str], label: str, value, low, high):
    if value is None:
        return
    if value < low or value > high:
        errors.append(f"{label} should be between {low} and {high}.")


def _validate_record_values(record: PatientHealthRecord) -> list[str]:
    errors: list[str] = []

    _validate_range(errors, "Age", record.age_years, 0, 120)
    _validate_range(errors, "Sex code", record.sex_code, 1, 2)

    _validate_range(errors, "Height", record.height_cm, 80, 250)
    _validate_range(errors, "Weight", record.weight_kg, 20, 300)
    _validate_range(errors, "BMI", record.bmi, 10, 80)
    _validate_range(errors, "Waist circumference", record.waist_cm, 40, 200)

    _validate_range(errors, "HbA1c", record.hba1c, 3, 20)

    _validate_range(errors, "Systolic blood pressure", record.systolic_bp, 70, 260)
    _validate_range(errors, "Diastolic blood pressure", record.diastolic_bp, 40, 160)
    _validate_range(errors, "Blood pressure reading count", record.bp_readings_count, 1, 20)

    _validate_range(errors, "Calories", record.calories_kcal_day, 300, 8000)
    _validate_range(errors, "Sodium", record.sodium_mg_day, 0, 15000)
    _validate_range(errors, "Sugar", record.sugar_g_day, 0, 500)
    _validate_range(errors, "Fibre", record.fibre_g_day, 0, 100)
    _validate_range(errors, "Saturated fat", record.saturated_fat_g_day, 0, 250)
    _validate_range(errors, "Alcohol", record.alcohol_g_day, 0, 300)

    _validate_range(errors, "Physical activity", record.mvpa_minutes_week, 0, 3000)
    _validate_range(errors, "Sedentary time", record.sedentary_hours_day, 0, 24)

    _validate_range(errors, "Total cholesterol", record.total_cholesterol_mg_dl, 50, 500)
    _validate_range(errors, "HDL cholesterol", record.hdl_mg_dl, 10, 150)

    return errors


def _populate_record_from_post(record: PatientHealthRecord, post_data):
    recorded_at = post_data.get("recorded_at", "").strip()
    if recorded_at:
        parsed_recorded_at = parse_date(recorded_at)
        if parsed_recorded_at is None:
            raise ValueError("Invalid recorded date.")
        record.recorded_at = parsed_recorded_at
    else:
        record.recorded_at = timezone.localdate()

    record.source = post_data.get("source", PatientHealthRecord.Source.PATIENT)
    record.label = post_data.get("label", "").strip() or "Health record"

    record.age_years = _int_or_none(post_data.get("age_years", ""))
    record.sex_code = _int_or_none(post_data.get("sex_code", ""))

    record.height_cm = _float_or_none(post_data.get("height_cm", ""))
    record.weight_kg = _float_or_none(post_data.get("weight_kg", ""))
    record.bmi = _float_or_none(post_data.get("bmi", ""))
    record.waist_cm = _float_or_none(post_data.get("waist_cm", ""))

    if not record.bmi:
        record.bmi = record.calculate_bmi_if_possible()

    record.hba1c = _float_or_none(post_data.get("hba1c", ""))

    record.systolic_bp = _float_or_none(post_data.get("systolic_bp", ""))
    record.diastolic_bp = _float_or_none(post_data.get("diastolic_bp", ""))
    record.bp_readings_count = _int_or_none(post_data.get("bp_readings_count", ""))

    record.diet_day_reliable = bool(post_data.get("diet_day_reliable"))
    record.calories_kcal_day = _float_or_none(post_data.get("calories_kcal_day", ""))
    record.sodium_mg_day = _float_or_none(post_data.get("sodium_mg_day", ""))
    record.sugar_g_day = _float_or_none(post_data.get("sugar_g_day", ""))
    record.fibre_g_day = _float_or_none(post_data.get("fibre_g_day", ""))
    record.saturated_fat_g_day = _float_or_none(post_data.get("saturated_fat_g_day", ""))
    record.alcohol_g_day = _float_or_none(post_data.get("alcohol_g_day", ""))

    record.mvpa_minutes_week = _int_or_none(post_data.get("mvpa_minutes_week", ""))
    record.sedentary_hours_day = _float_or_none(post_data.get("sedentary_hours_day", ""))

    record.current_smoker = bool(post_data.get("current_smoker"))
    record.alcohol_over_guideline = bool(post_data.get("alcohol_over_guideline"))

    record.total_cholesterol_mg_dl = _float_or_none(
        post_data.get("total_cholesterol_mg_dl", "")
    )
    record.hdl_mg_dl = _float_or_none(post_data.get("hdl_mg_dl", ""))

    record.known_prediabetes = bool(post_data.get("known_prediabetes"))
    record.known_severe_hyperglycemia = bool(post_data.get("known_severe_hyperglycemia"))
    record.known_hypertensive_crisis = bool(post_data.get("known_hypertensive_crisis"))

    record.notes = post_data.get("notes", "").strip()

    return record


def _render_form(request, record: PatientHealthRecord, mode: str):
    return render(
        request,
        "user/health_data_form.html",
        {
            "record": record,
            "sources": PatientHealthRecord.Source.choices,
            "mode": mode,
        },
    )


@role_required(User.Role.ADMIN, User.Role.CLINICIAN, User.Role.USER)
def health_data(request):
    records_qs = (
        PatientHealthRecord.objects
        .filter(user=request.user)
        .order_by("-recorded_at", "-created_at")
    )

    latest_record = records_qs.first()
    record_count = records_qs.count()

    current_payload = build_current_health_feature_payload(request.user)

    important_fields = [
        "RIDAGEYR",
        "RIAGENDR",
        "BMXBMI",
        "BMXWAIST",
        "LBXGH",
        "SBP_mean",
        "DBP_mean",
        "DR1TSODI",
        "DR1TSUGR",
        "DR1TFIBE",
        "Any_MVPA_flag",
        "Current_smoker_flag",
        "Alcohol_over_guideline_flag",
        "TotalChol_mg_dL",
        "HDL_mg_dL",
    ]

    important_completed = sum(
        1
        for key in important_fields
        if current_payload.get(key) not in [None, ""]
    )
    completeness_pct = round((important_completed / len(important_fields)) * 100)

    return render(
        request,
        "user/health_data.html",
        {
            "records": records_qs[:10],
            "latest_record": latest_record,
            "record_count": record_count,
            "current_payload": current_payload,
            "important_completed": important_completed,
            "important_total": len(important_fields),
            "completeness_pct": completeness_pct,
        },
    )


@role_required(User.Role.ADMIN, User.Role.CLINICIAN, User.Role.USER)
def health_data_add(request):
    record = PatientHealthRecord(
        user=request.user,
        recorded_at=timezone.localdate(),
        diet_day_reliable=True,
    )

    if request.method == "POST":
        try:
            record = _populate_record_from_post(record, request.POST)
            errors = _validate_record_values(record)

            if errors:
                for error in errors:
                    messages.error(request, error)
                return _render_form(request, record, mode="add")

            record.save()
            messages.success(request, "Health record saved.")
            return redirect("user:health_data_detail", record_id=record.id)

        except ValueError:
            messages.error(request, "Please check the numbers and date entered.")
            return _render_form(request, record, mode="add")

    return _render_form(request, record, mode="add")


@role_required(User.Role.ADMIN, User.Role.CLINICIAN, User.Role.USER)
def health_data_edit(request, record_id: int):
    record = get_object_or_404(PatientHealthRecord, id=record_id, user=request.user)

    if request.method == "POST":
        try:
            record = _populate_record_from_post(record, request.POST)
            errors = _validate_record_values(record)

            if errors:
                for error in errors:
                    messages.error(request, error)
                return _render_form(request, record, mode="edit")

            record.save()
            messages.success(request, "Health record updated.")
            return redirect("user:health_data_detail", record_id=record.id)

        except ValueError:
            messages.error(request, "Please check the numbers and date entered.")
            return _render_form(request, record, mode="edit")

    return _render_form(request, record, mode="edit")


@role_required(User.Role.ADMIN, User.Role.CLINICIAN, User.Role.USER)
def health_data_detail(request, record_id: int):
    record = get_object_or_404(PatientHealthRecord, id=record_id, user=request.user)

    return render(
        request,
        "user/health_data_detail.html",
        {
            "record": record,
        },
    )


@role_required(User.Role.ADMIN, User.Role.CLINICIAN, User.Role.USER)
def health_data_delete(request, record_id: int):
    record = get_object_or_404(PatientHealthRecord, id=record_id, user=request.user)

    if request.method == "POST":
        record.delete()
        messages.success(request, "Health record deleted.")
        return redirect("user:health_data")

    return render(
        request,
        "user/health_data_delete.html",
        {
            "record": record,
        },
    )
