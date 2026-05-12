from __future__ import annotations

from django.contrib import messages
from django.http import HttpResponseForbidden
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone

from apps.accounts.models import User
from apps.accounts.permissions import role_required
from apps.profiles.models import ClinicianHealthRecord
from apps.profiles.services import clinician_can_access_patient
from apps.recommendations.models import RecommendationItem, RecommendationRun
from apps.recommendations.services.live_pipeline import run_live_patient_pipeline
from hybrid_engine.engine import generate_recommendations


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


def _validate_record_values(record: ClinicianHealthRecord) -> list[str]:
    errors: list[str] = []

    _validate_range(errors, "Age", record.age_years, 0, 120)
    _validate_range(errors, "Sex code", record.sex_code, 1, 2)

    for label, value in [
        ("BPXSY1", record.bpxsy1),
        ("BPXSY2", record.bpxsy2),
        ("BPXSY3", record.bpxsy3),
        ("BPXSY4", record.bpxsy4),
        ("Systolic blood pressure override", record.systolic_bp),
    ]:
        _validate_range(errors, label, value, 70, 260)

    for label, value in [
        ("BPXDI1", record.bpxdi1),
        ("BPXDI2", record.bpxdi2),
        ("BPXDI3", record.bpxdi3),
        ("BPXDI4", record.bpxdi4),
        ("Diastolic blood pressure override", record.diastolic_bp),
    ]:
        _validate_range(errors, label, value, 40, 160)

    _validate_range(errors, "Pulse", record.pulse_bpm, 30, 220)
    _validate_range(errors, "Blood pressure reading count", record.bp_readings_count, 1, 20)

    _validate_range(errors, "Height", record.height_cm, 80, 250)
    _validate_range(errors, "Weight", record.weight_kg, 20, 300)
    _validate_range(errors, "BMI", record.bmi, 10, 80)
    _validate_range(errors, "Waist circumference", record.waist_cm, 40, 200)
    _validate_range(errors, "Hip circumference", record.hip_cm, 40, 220)

    _validate_range(errors, "HbA1c", record.hba1c, 3, 20)

    _validate_range(errors, "Total cholesterol", record.total_cholesterol_mg_dl, 50, 500)
    _validate_range(errors, "HDL cholesterol", record.hdl_mg_dl, 10, 150)

    _validate_range(errors, "Calories", record.calories_kcal_day, 300, 8000)
    _validate_range(errors, "Protein", record.protein_g_day, 0, 400)
    _validate_range(errors, "Carbohydrate", record.carbohydrate_g_day, 0, 1000)
    _validate_range(errors, "Total fat", record.total_fat_g_day, 0, 500)
    _validate_range(errors, "Saturated fat", record.saturated_fat_g_day, 0, 250)
    _validate_range(errors, "Sugar", record.sugar_g_day, 0, 500)
    _validate_range(errors, "Fibre", record.fibre_g_day, 0, 100)
    _validate_range(errors, "Sodium", record.sodium_mg_day, 0, 15000)
    _validate_range(errors, "Potassium", record.potassium_mg_day, 0, 15000)
    _validate_range(errors, "Alcohol", record.alcohol_g_day, 0, 300)

    _validate_range(errors, "Physical activity", record.mvpa_minutes_week, 0, 3000)
    _validate_range(errors, "Sedentary time", record.sedentary_hours_day, 0, 24)

    return errors


def _populate_record_from_post(record: ClinicianHealthRecord, request) -> ClinicianHealthRecord:
    recorded_at = request.POST.get("recorded_at", "").strip()
    record.recorded_at = recorded_at or timezone.localdate()

    record.source = request.POST.get("source", ClinicianHealthRecord.Source.CLINIC)
    record.label = request.POST.get("label", "").strip() or "Clinician-entered health record"

    record.age_years = _int_or_none(request.POST.get("age_years", ""))
    record.sex_code = _int_or_none(request.POST.get("sex_code", ""))

    record.bpxsy1 = _float_or_none(request.POST.get("bpxsy1", ""))
    record.bpxsy2 = _float_or_none(request.POST.get("bpxsy2", ""))
    record.bpxsy3 = _float_or_none(request.POST.get("bpxsy3", ""))
    record.bpxsy4 = _float_or_none(request.POST.get("bpxsy4", ""))

    record.bpxdi1 = _float_or_none(request.POST.get("bpxdi1", ""))
    record.bpxdi2 = _float_or_none(request.POST.get("bpxdi2", ""))
    record.bpxdi3 = _float_or_none(request.POST.get("bpxdi3", ""))
    record.bpxdi4 = _float_or_none(request.POST.get("bpxdi4", ""))

    record.pulse_bpm = _float_or_none(request.POST.get("pulse_bpm", ""))

    record.systolic_bp = _float_or_none(request.POST.get("systolic_bp", "")) or record.mean_sbp_from_readings()
    record.diastolic_bp = _float_or_none(request.POST.get("diastolic_bp", "")) or record.mean_dbp_from_readings()
    record.bp_readings_count = _int_or_none(request.POST.get("bp_readings_count", "")) or record.bp_reading_count_from_readings()

    record.height_cm = _float_or_none(request.POST.get("height_cm", ""))
    record.weight_kg = _float_or_none(request.POST.get("weight_kg", ""))
    record.bmi = _float_or_none(request.POST.get("bmi", ""))
    record.waist_cm = _float_or_none(request.POST.get("waist_cm", ""))
    record.hip_cm = _float_or_none(request.POST.get("hip_cm", ""))

    if record.bmi is None:
        record.bmi = record.calculate_bmi_if_possible()

    record.hba1c = _float_or_none(request.POST.get("hba1c", ""))

    record.total_cholesterol_mg_dl = _float_or_none(request.POST.get("total_cholesterol_mg_dl", ""))
    record.hdl_mg_dl = _float_or_none(request.POST.get("hdl_mg_dl", ""))

    record.diet_day_reliable = bool(request.POST.get("diet_day_reliable"))
    record.calories_kcal_day = _float_or_none(request.POST.get("calories_kcal_day", ""))
    record.protein_g_day = _float_or_none(request.POST.get("protein_g_day", ""))
    record.carbohydrate_g_day = _float_or_none(request.POST.get("carbohydrate_g_day", ""))
    record.total_fat_g_day = _float_or_none(request.POST.get("total_fat_g_day", ""))
    record.saturated_fat_g_day = _float_or_none(request.POST.get("saturated_fat_g_day", ""))
    record.sugar_g_day = _float_or_none(request.POST.get("sugar_g_day", ""))
    record.fibre_g_day = _float_or_none(request.POST.get("fibre_g_day", ""))
    record.sodium_mg_day = _float_or_none(request.POST.get("sodium_mg_day", ""))
    record.potassium_mg_day = _float_or_none(request.POST.get("potassium_mg_day", ""))
    record.alcohol_g_day = _float_or_none(request.POST.get("alcohol_g_day", ""))

    record.smoking_status = request.POST.get("smoking_status", "").strip()
    record.current_smoker = bool(request.POST.get("current_smoker")) or record.smoking_status == "current"
    record.mvpa_minutes_week = _int_or_none(request.POST.get("mvpa_minutes_week", ""))
    record.vigorous_activity = bool(request.POST.get("vigorous_activity"))
    record.moderate_activity = bool(request.POST.get("moderate_activity"))
    record.sedentary_hours_day = _float_or_none(request.POST.get("sedentary_hours_day", ""))
    record.alcohol_over_guideline = bool(request.POST.get("alcohol_over_guideline"))

    record.known_prediabetes = bool(request.POST.get("known_prediabetes"))
    record.known_severe_hyperglycemia = bool(request.POST.get("known_severe_hyperglycemia"))
    record.known_hypertensive_crisis = bool(request.POST.get("known_hypertensive_crisis"))

    record.notes = request.POST.get("notes", "").strip()

    return record


@role_required(User.Role.ADMIN, User.Role.CLINICIAN)
def clinician_health_data_add(request, patient_id: int):
    patient = get_object_or_404(User, id=patient_id)

    if not clinician_can_access_patient(request.user, patient):
        return HttpResponseForbidden("Forbidden: patient not assigned.")

    if request.method == "POST":
        record = ClinicianHealthRecord(
            patient=patient,
            clinician=request.user,
        )

        try:
            record = _populate_record_from_post(record, request)
            errors = _validate_record_values(record)

            if errors:
                for error in errors:
                    messages.error(request, error)

                return render(
                    request,
                    "clinician/health_data_form.html",
                    {
                        "patient": patient,
                        "record": record,
                        "sources": ClinicianHealthRecord.Source.choices,
                    },
                )

            record.save()
            messages.success(request, "Clinical health record saved for this patient.")
            return redirect("clinician:patient_detail", patient_id=patient.id)

        except ValueError:
            messages.error(request, "Please check the numbers and date entered.")

    else:
        record = ClinicianHealthRecord(
            patient=patient,
            clinician=request.user,
            recorded_at=timezone.localdate(),
            source=ClinicianHealthRecord.Source.CLINIC,
            label="Clinician-entered health record",
            diet_day_reliable=True,
        )

    return render(
        request,
        "clinician/health_data_form.html",
        {
            "patient": patient,
            "record": record,
            "sources": ClinicianHealthRecord.Source.choices,
        },
    )


@role_required(User.Role.ADMIN, User.Role.CLINICIAN)
def clinician_generate_patient_recommendations(request, patient_id: int):
    patient = get_object_or_404(User, id=patient_id)

    if not clinician_can_access_patient(request.user, patient):
        return HttpResponseForbidden("Forbidden: patient not assigned.")

    if request.method != "POST":
        return redirect("clinician:patient_detail", patient_id=patient.id)

    pipeline_mode = "live_single_patient_pipeline"
    pipeline_notes = ""
    recs = []

    try:
        live_result = run_live_patient_pipeline(patient)
        recs = live_result.recommendations
        pipeline_notes = (
            "Generated by clinician using the full live single-patient pipeline: "
            "rules engine, cold-start safety/context, ranking, ML reranking/fusion, "
            "bundling, and explainability. "
            f"Clinician: {request.user.username}. "
            f"Manifest: {live_result.manifest}"
        )

        messages.success(
            request,
            f"Recommendations generated for {patient.username} using the live pipeline.",
        )

    except Exception as exc:
        pipeline_mode = "reference_match_fallback"

        fallback_recs = generate_recommendations(
            user_context={
                "user_id": patient.id,
                "generated_by": request.user.id,
            }
        )

        recs = fallback_recs
        pipeline_notes = (
            "Clinician-triggered live single-patient pipeline failed and "
            "reference-matching fallback was used. "
            f"Clinician: {request.user.username}. "
            f"Failure: {type(exc).__name__}: {exc}"
        )

        messages.warning(
            request,
            "The full live pipeline could not complete, so the system used the safer reference-matching fallback.",
        )

    if not recs:
        messages.warning(
            request,
            "No recommendations were available. Add clinical health data and try again.",
        )
        return redirect("clinician:patient_detail", patient_id=patient.id)

    run = RecommendationRun.objects.create(
        user=patient,
        engine_version=pipeline_mode,
        notes=pipeline_notes,
    )

    for r in recs:
        RecommendationItem.objects.create(
            run=run,
            title=r.title,
            category=r.category,
            priority=r.priority,
            rationale=r.rationale,
            explanation=r.explanation,
            source=r.source[:32],
            confidence=r.confidence,
        )

    return redirect(
        "clinician:patient_run_detail",
        patient_id=patient.id,
        run_id=run.id,
    )
