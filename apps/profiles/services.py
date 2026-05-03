from __future__ import annotations

from datetime import date
from typing import Any

from django.utils import timezone

from apps.accounts.models import User
from .models import ClinicianPatientLink, PatientHealthRecord, PatientProfile


def clinician_can_access_patient(clinician: User, patient: User) -> bool:
    if clinician.role == User.Role.ADMIN:
        return True
    return ClinicianPatientLink.objects.filter(clinician=clinician, patient=patient).exists()


def _age_from_dob(dob: date | None) -> int | None:
    if not dob:
        return None

    today = timezone.localdate()
    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    return age if age >= 0 else None


def _sex_to_nhanes_code(sex: str | None) -> int | None:
    text = (sex or "").strip().lower()

    if text in {"male", "m", "man", "1"}:
        return 1

    if text in {"female", "f", "woman", "2"}:
        return 2

    return None


def latest_non_empty(records: list[PatientHealthRecord], attr: str):
    for record in records:
        value = getattr(record, attr, None)
        if value not in [None, ""]:
            return value
    return None


def _flag(value: bool | None) -> int:
    return int(bool(value))


def _hba1c_category(hba1c: float | None) -> tuple[int, int, int]:
    """
    Returns:
    Prediabetes, Diabetes, Severe_hyperglycemia_flag

    This is a deterministic support mapping only. The patient UI should not
    present it as diagnosis.
    """
    if hba1c is None:
        return 0, 0, 0

    prediabetes = int(5.7 <= hba1c < 6.5)
    diabetes = int(hba1c >= 6.5)
    severe = int(hba1c >= 9.0)

    return prediabetes, diabetes, severe


def _bp_flags(sbp: float | None, dbp: float | None) -> tuple[int, int]:
    """
    Returns:
    Hypertension, Hypertensive_crisis_flag
    """
    if sbp is None and dbp is None:
        return 0, 0

    sbp_val = sbp or 0
    dbp_val = dbp or 0

    hypertension = int(sbp_val >= 130 or dbp_val >= 80)
    crisis = int(sbp_val >= 180 or dbp_val >= 120)

    return hypertension, crisis


def _bmi_flags(bmi: float | None) -> tuple[int, int]:
    """
    Returns:
    Severe_obesity_flag, obesity_flag_proxy
    """
    if bmi is None:
        return 0, 0

    severe_obesity = int(bmi >= 40)
    obesity = int(bmi >= 30)

    return severe_obesity, obesity


def _cholesterol_flags(total_chol: float | None, hdl: float | None) -> tuple[int, int]:
    """
    Returns:
    TC_high_flag, HDL_low_flag
    """
    tc_high = int(total_chol is not None and total_chol >= 240)
    hdl_low = int(hdl is not None and hdl < 40)

    return tc_high, hdl_low


def _satfat_over_10pct_flag(satfat_g: float | None, calories: float | None) -> int:
    if satfat_g is None or calories is None or calories <= 0:
        return 0

    # 1g fat = 9 kcal
    satfat_pct = (satfat_g * 9 / calories) * 100
    return int(satfat_pct > 10)


def _severity_band(score: float) -> str:
    if score >= 7:
        return "high"
    if score >= 4:
        return "moderate"
    if score > 0:
        return "low"
    return "minimal"


def build_current_health_feature_payload(user: User) -> dict[str, Any]:
    """
    Build a current feature payload from the user's health record timeline.

    Logic:
    - Use the latest non-empty value for each patient-entered measurement.
    - Stable context comes from PatientProfile.
    - Derived flags are computed deterministically.
    - System-generated recommendation/candidate/collaborative fields are not created here.
    """
    profile, _ = PatientProfile.objects.get_or_create(user=user)
    records = list(
        PatientHealthRecord.objects
        .filter(user=user)
        .order_by("-recorded_at", "-created_at")
    )

    latest_record = records[0] if records else None

    age = latest_non_empty(records, "age_years") or _age_from_dob(profile.date_of_birth)
    sex_code = latest_non_empty(records, "sex_code") or _sex_to_nhanes_code(profile.sex)

    height_cm = latest_non_empty(records, "height_cm")
    weight_kg = latest_non_empty(records, "weight_kg")
    bmi = latest_non_empty(records, "bmi")

    if bmi is None and height_cm and weight_kg:
        height_m = height_cm / 100
        if height_m > 0:
            bmi = round(weight_kg / (height_m * height_m), 1)

    waist_cm = latest_non_empty(records, "waist_cm")
    hba1c = latest_non_empty(records, "hba1c")

    sbp = latest_non_empty(records, "systolic_bp")
    dbp = latest_non_empty(records, "diastolic_bp")
    bp_n_used = latest_non_empty(records, "bp_readings_count") or (1 if sbp or dbp else None)

    calories = latest_non_empty(records, "calories_kcal_day")
    sodium = latest_non_empty(records, "sodium_mg_day")
    sugar = latest_non_empty(records, "sugar_g_day")
    fibre = latest_non_empty(records, "fibre_g_day")
    satfat = latest_non_empty(records, "saturated_fat_g_day")
    alcohol_g = latest_non_empty(records, "alcohol_g_day")

    mvpa_minutes_week = latest_non_empty(records, "mvpa_minutes_week")
    sedentary_hours_day = latest_non_empty(records, "sedentary_hours_day")

    total_chol = latest_non_empty(records, "total_cholesterol_mg_dl")
    hdl = latest_non_empty(records, "hdl_mg_dl")

    current_smoker = bool(latest_non_empty(records, "current_smoker"))
    alcohol_over_guideline = bool(latest_non_empty(records, "alcohol_over_guideline"))

    profile_diabetes = int(profile.has_dm)
    profile_hypertension = int(profile.has_htn)

    prediabetes_from_hba1c, diabetes_from_hba1c, severe_hyperglycemia_from_hba1c = _hba1c_category(hba1c)
    hypertension_from_bp, crisis_from_bp = _bp_flags(sbp, dbp)
    severe_obesity_flag, obesity_flag_proxy = _bmi_flags(bmi)
    tc_high_flag, hdl_low_flag = _cholesterol_flags(total_chol, hdl)

    diabetes = max(profile_diabetes, diabetes_from_hba1c)
    hypertension = max(profile_hypertension, hypertension_from_bp)
    prediabetes = prediabetes_from_hba1c if not diabetes else 0

    any_mvpa_flag = int((mvpa_minutes_week or 0) >= 150)
    sedentary_high_flag = int((sedentary_hours_day or 0) >= 8)

    sodium_over_2300_flag = int(sodium is not None and sodium > 2300)
    sodium_over_1500_flag = int(sodium is not None and sodium > 1500)
    satfat_over_10pct_flag = _satfat_over_10pct_flag(satfat, calories)

    diet_reliable = _flag(latest_record.diet_day_reliable) if latest_record else 0

    # Simple deterministic severity components.
    htn_severity = 0
    if crisis_from_bp:
        htn_severity = 3
    elif hypertension:
        htn_severity = 2

    dm_severity = 0
    if severe_hyperglycemia_from_hba1c:
        dm_severity = 3
    elif diabetes:
        dm_severity = 2
    elif prediabetes:
        dm_severity = 1

    obesity_severity = 0
    if bmi is not None:
        if bmi >= 40:
            obesity_severity = 3
        elif bmi >= 30:
            obesity_severity = 2
        elif bmi >= 25:
            obesity_severity = 1

    lifestyle_risk = (
        int(current_smoker)
        + int(alcohol_over_guideline)
        + int(any_mvpa_flag == 0)
        + sedentary_high_flag
        + sodium_over_2300_flag
    )

    cardiometabolic_score = (
        htn_severity
        + dm_severity
        + obesity_severity
        + int(tc_high_flag)
        + int(hdl_low_flag)
        + int(lifestyle_risk >= 2)
    )

    comorbidity = int(sum([hypertension, diabetes, obesity_flag_proxy, tc_high_flag]) >= 2)

    core_fields = [sbp, dbp, hba1c, bmi, sodium]
    available_core_fields = sum(1 for v in core_fields if v is not None)
    data_completeness_score = round(available_core_fields / len(core_fields), 3)

    payload = {
        # Dataset/patient-level columns
        "RIDAGEYR": age,
        "RIAGENDR": sex_code,
        "BMXBMI": bmi,
        "BMXWAIST": waist_cm,
        "LBXGH": hba1c,
        "SBP_mean": sbp,
        "DBP_mean": dbp,
        "BP_n_used": bp_n_used,

        # Diet/lifestyle
        "DietDay1_reliable_flag": diet_reliable,
        "DR1TKCAL": calories,
        "DR1TSODI": sodium,
        "DR1TSUGR": sugar,
        "DR1TFIBE": fibre,
        "DR1TSFAT": satfat,
        "DR1TALCO": alcohol_g,
        "Any_MVPA_flag": any_mvpa_flag,
        "Current_smoker_flag": int(current_smoker),
        "Alcohol_over_guideline_flag": int(alcohol_over_guideline),
        "Sedentary_high_flag": sedentary_high_flag,

        # Lipids
        "TotalChol_mg_dL": total_chol,
        "HDL_mg_dL": hdl,

        # Derived clinical flags
        "Hypertension": hypertension,
        "Hypertensive_crisis_flag": max(crisis_from_bp, int(bool(latest_non_empty(records, "known_hypertensive_crisis")))),
        "Diabetes": diabetes,
        "Prediabetes": max(prediabetes, int(bool(latest_non_empty(records, "known_prediabetes")))),
        "Severe_hyperglycemia_flag": max(
            severe_hyperglycemia_from_hba1c,
            int(bool(latest_non_empty(records, "known_severe_hyperglycemia"))),
        ),
        "Severe_obesity_flag": severe_obesity_flag,
        "Sodium_over_2300_flag": sodium_over_2300_flag,
        "Sodium_over_1500_flag": sodium_over_1500_flag,
        "SatFat_over_10pct_flag": satfat_over_10pct_flag,
        "HDL_low_flag": hdl_low_flag,
        "TC_high_flag": tc_high_flag,
        "Comorbidity": comorbidity,

        # Deterministic severity/profile features
        "HTN_severity_score": htn_severity,
        "DM_severity_score": dm_severity,
        "Obesity_severity_score": obesity_severity,
        "Lifestyle_risk_score": lifestyle_risk,
        "Cardiometabolic_severity_score": cardiometabolic_score,
        "Severity_band": _severity_band(cardiometabolic_score),
        "data_completeness_score": data_completeness_score,

        # Profile-style features used by active ML model
        "profile_age_group": _profile_age_group(age),
        "profile_cardiometabolic_risk": _profile_cardiometabolic_risk(cardiometabolic_score),
        "profile_glycaemic_status": _profile_glycaemic_status(diabetes, prediabetes, hba1c),
        "profile_bp_status": _profile_bp_status(hypertension, crisis_from_bp),
        "profile_weight_status": _profile_weight_status(bmi),
        "profile_version": "patient_entered_v1",
        "profile_data_completeness_score": data_completeness_score,
        "profile_current_smoker_flag": int(current_smoker),
        "profile_inactive_flag": int(any_mvpa_flag == 0),
        "profile_high_alcohol_risk_flag": int(alcohol_over_guideline),
        "profile_poor_diet_signal_flag": int(
            sodium_over_2300_flag or satfat_over_10pct_flag or (fibre is not None and fibre < 25)
        ),

        # Record metadata for audit/debug
        "_record_count": len(records),
        "_latest_record_id": latest_record.id if latest_record else None,
        "_latest_recorded_at": latest_record.recorded_at.isoformat() if latest_record else None,
    }

    return payload


def _profile_age_group(age: int | None) -> str:
    if age is None:
        return "unknown"
    if age < 40:
        return "18_39"
    if age < 60:
        return "40_59"
    return "60_plus"


def _profile_cardiometabolic_risk(score: float) -> str:
    if score >= 7:
        return "high"
    if score >= 4:
        return "moderate"
    if score > 0:
        return "low"
    return "unknown"


def _profile_glycaemic_status(diabetes: int, prediabetes: int, hba1c: float | None) -> str:
    if diabetes:
        return "diabetes"
    if prediabetes:
        return "prediabetes"
    if hba1c is not None:
        return "normoglycaemic"
    return "unknown"


def _profile_bp_status(hypertension: int, crisis: int) -> str:
    if crisis:
        return "crisis_range"
    if hypertension:
        return "elevated"
    return "normal_or_unknown"


def _profile_weight_status(bmi: float | None) -> str:
    if bmi is None:
        return "unknown"
    if bmi >= 30:
        return "obesity"
    if bmi >= 25:
        return "overweight"
    return "normal_or_underweight"
