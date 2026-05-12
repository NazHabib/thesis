from __future__ import annotations

from datetime import date
from typing import Any

from django.utils import timezone

from apps.accounts.models import User
from .models import ClinicianHealthRecord, ClinicianPatientLink, PatientHealthRecord, PatientProfile


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
    
def latest_non_empty_from_records(records: list, attr: str):
    for record in records:
        value = getattr(record, attr, None)
        if value not in [None, ""]:
            return value
    return None


def _mean_non_empty(*values):
    nums = []
    for value in values:
        if value in [None, ""]:
            continue
        try:
            nums.append(float(value))
        except (TypeError, ValueError):
            continue

    if not nums:
        return None

    return round(sum(nums) / len(nums), 2)


def _combined_health_records(user: User) -> tuple[list[ClinicianHealthRecord], list[PatientHealthRecord], list]:
    clinician_records = list(
        ClinicianHealthRecord.objects
        .filter(patient=user)
        .order_by("-recorded_at", "-created_at")
    )

    patient_records = list(
        PatientHealthRecord.objects
        .filter(user=user)
        .order_by("-recorded_at", "-created_at")
    )

    # Priority decision:
    # clinician/lab/clinic records are trusted first, then patient/family/home records.
    combined_records = clinician_records + patient_records

    return clinician_records, patient_records, combined_records

def any_true(records: list[PatientHealthRecord], attr: str) -> bool:
    return any(bool(getattr(record, attr, False)) for record in records)

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

class CurrentHealthFeatureBuilder:
    """
    Builds the current pipeline-compatible health feature payload for one patient.

    Priority order:
    1. ClinicianHealthRecord values
    2. PatientHealthRecord values
    3. PatientProfile fallback values

    This keeps the public pipeline input stable while allowing clinician-entered
    clinical records to provide richer and higher-trust data than patient/family
    self-reported records.
    """

    def __init__(self, user: User):
        if user is None:
            raise ValueError("CurrentHealthFeatureBuilder requires a valid user.")

        self.user = user
        self.profile, _ = PatientProfile.objects.get_or_create(user=user)

        self.clinician_records = list(
            ClinicianHealthRecord.objects
            .filter(patient=user)
            .order_by("-recorded_at", "-created_at")
        )

        self.patient_records = list(
            PatientHealthRecord.objects
            .filter(user=user)
            .order_by("-recorded_at", "-created_at")
        )

        self.records = self.clinician_records + self.patient_records

        self.latest_clinician_record = (
            self.clinician_records[0] if self.clinician_records else None
        )
        self.latest_patient_record = (
            self.patient_records[0] if self.patient_records else None
        )
        self.latest_record = self.latest_clinician_record or self.latest_patient_record

    def latest(self, attr: str):
        for record in self.records:
            value = getattr(record, attr, None)
            if value not in [None, ""]:
                return value
        return None

    def any_true(self, attr: str) -> bool:
        return any(bool(getattr(record, attr, False)) for record in self.records)

    @staticmethod
    def mean_non_empty(*values):
        nums = []
        for value in values:
            if value in [None, ""]:
                continue
            try:
                nums.append(float(value))
            except (TypeError, ValueError):
                continue

        if not nums:
            return None

        return round(sum(nums) / len(nums), 2)

    @staticmethod
    def count_non_empty(*values) -> int:
        return sum(1 for value in values if value not in [None, ""])

    def build(self) -> dict[str, Any]:
        age = self.latest("age_years") or _age_from_dob(self.profile.date_of_birth)
        sex_code = self.latest("sex_code") or _sex_to_nhanes_code(self.profile.sex)

        height_cm = self.latest("height_cm")
        weight_kg = self.latest("weight_kg")
        bmi = self.latest("bmi")

        if bmi is None and height_cm and weight_kg:
            height_m = height_cm / 100
            if height_m > 0:
                bmi = round(weight_kg / (height_m * height_m), 1)

        waist_cm = self.latest("waist_cm")
        hip_cm = self.latest("hip_cm")
        hba1c = self.latest("hba1c")

        sbp1 = self.latest("bpxsy1")
        sbp2 = self.latest("bpxsy2")
        sbp3 = self.latest("bpxsy3")
        sbp4 = self.latest("bpxsy4")

        dbp1 = self.latest("bpxdi1")
        dbp2 = self.latest("bpxdi2")
        dbp3 = self.latest("bpxdi3")
        dbp4 = self.latest("bpxdi4")

        sbp = self.latest("systolic_bp") or self.mean_non_empty(sbp1, sbp2, sbp3, sbp4)
        dbp = self.latest("diastolic_bp") or self.mean_non_empty(dbp1, dbp2, dbp3, dbp4)

        bp_n_used = (
            self.latest("bp_readings_count")
            or min(
                self.count_non_empty(sbp1, sbp2, sbp3, sbp4),
                self.count_non_empty(dbp1, dbp2, dbp3, dbp4),
            )
            or (1 if sbp or dbp else None)
        )

        calories = self.latest("calories_kcal_day")
        sodium = self.latest("sodium_mg_day")
        sugar = self.latest("sugar_g_day")
        fibre = self.latest("fibre_g_day")
        satfat = self.latest("saturated_fat_g_day")
        alcohol_g = self.latest("alcohol_g_day")

        protein = self.latest("protein_g_day")
        carbohydrate = self.latest("carbohydrate_g_day")
        total_fat = self.latest("total_fat_g_day")
        potassium = self.latest("potassium_mg_day")

        mvpa_minutes_week = self.latest("mvpa_minutes_week")
        sedentary_hours_day = self.latest("sedentary_hours_day")

        total_chol = self.latest("total_cholesterol_mg_dl")
        hdl = self.latest("hdl_mg_dl")

        smoking_status = self.latest("smoking_status")
        current_smoker = bool(self.latest("current_smoker")) or smoking_status == "current"
        alcohol_over_guideline = bool(self.latest("alcohol_over_guideline"))

        vigorous_activity = self.latest("vigorous_activity")
        moderate_activity = self.latest("moderate_activity")

        profile_diabetes = int(self.profile.has_dm)
        profile_hypertension = int(self.profile.has_htn)

        prediabetes_from_hba1c, diabetes_from_hba1c, severe_hyperglycemia_from_hba1c = _hba1c_category(hba1c)
        hypertension_from_bp, crisis_from_bp = _bp_flags(sbp, dbp)
        severe_obesity_flag, obesity_flag_proxy = _bmi_flags(bmi)
        tc_high_flag, hdl_low_flag = _cholesterol_flags(total_chol, hdl)

        diabetes = max(profile_diabetes, diabetes_from_hba1c)
        hypertension = max(profile_hypertension, hypertension_from_bp)

        known_prediabetes = int(self.any_true("known_prediabetes"))
        known_severe_hyperglycemia = int(self.any_true("known_severe_hyperglycemia"))
        known_hypertensive_crisis = int(self.any_true("known_hypertensive_crisis"))

        prediabetes = 0 if diabetes else max(prediabetes_from_hba1c, known_prediabetes)

        any_mvpa_flag = (
            int(mvpa_minutes_week >= 150)
            if mvpa_minutes_week is not None
            else None
        )

        if vigorous_activity is not None or moderate_activity is not None:
            any_mvpa_flag = int(bool(vigorous_activity) or bool(moderate_activity))

        inactive_signal_flag = (
            int(mvpa_minutes_week < 150)
            if mvpa_minutes_week is not None
            else 0
        )

        sedentary_high_flag = (
            int(sedentary_hours_day >= 8)
            if sedentary_hours_day is not None
            else 0
        )

        sodium_over_2300_flag = int(sodium is not None and sodium > 2300)
        sodium_over_1500_flag = int(sodium is not None and sodium > 1500)
        satfat_over_10pct_flag = _satfat_over_10pct_flag(satfat, calories)

        diet_reliable = _flag(self.latest_record.diet_day_reliable) if self.latest_record else 0

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
            + inactive_signal_flag
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
            "RIDAGEYR": age,
            "RIAGENDR": sex_code,

            "BPXSY1": sbp1,
            "BPXSY2": sbp2,
            "BPXSY3": sbp3,
            "BPXSY4": sbp4,
            "BPXDI1": dbp1,
            "BPXDI2": dbp2,
            "BPXDI3": dbp3,
            "BPXDI4": dbp4,
            "BPXPLS": self.latest("pulse_bpm"),

            "SBP_mean": sbp,
            "DBP_mean": dbp,
            "BP_n_used": bp_n_used,

            "BMXHT": height_cm,
            "BMXWT": weight_kg,
            "BMXBMI": bmi,
            "BMXWAIST": waist_cm,
            "BMXHIP": hip_cm,

            "LBXGH": hba1c,

            "DietDay1_reliable_flag": diet_reliable,
            "DR1TKCAL": calories,
            "DR1TPROT": protein,
            "DR1TCARB": carbohydrate,
            "DR1TTFAT": total_fat,
            "DR1TSFAT": satfat,
            "DR1TSUGR": sugar,
            "DR1TFIBE": fibre,
            "DR1TSODI": sodium,
            "DR1TPOTA": potassium,
            "DR1TALCO": alcohol_g,

            "Any_MVPA_flag": any_mvpa_flag,
            "Vigorous_activity_flag": int(bool(vigorous_activity)) if vigorous_activity is not None else 0,
            "Moderate_activity_flag": int(bool(moderate_activity)) if moderate_activity is not None else 0,
            "Current_smoker_flag": int(current_smoker),
            "Smoking_status": smoking_status or ("current" if current_smoker else "not_current_or_unknown"),
            "Alcohol_over_guideline_flag": int(alcohol_over_guideline),
            "Sedentary_high_flag": sedentary_high_flag,

            "TotalChol_mg_dL": total_chol,
            "HDL_mg_dL": hdl,

            "Waist_to_hip_ratio": round(waist_cm / hip_cm, 3) if waist_cm and hip_cm else None,
            "Waist_to_height_ratio": round(waist_cm / height_cm, 3) if waist_cm and height_cm else None,
            "Potassium_to_Sodium_ratio": round(potassium / sodium, 3) if potassium and sodium else None,
            "Alcohol_drinks_equiv_day1": round(alcohol_g / 14, 2) if alcohol_g is not None else None,

            "Hypertension": hypertension,
            "Hypertensive_crisis_flag": max(crisis_from_bp, known_hypertensive_crisis),
            "Diabetes": diabetes,
            "Prediabetes": prediabetes,
            "Severe_hyperglycemia_flag": max(
                severe_hyperglycemia_from_hba1c,
                known_severe_hyperglycemia,
            ),
            "Severe_obesity_flag": severe_obesity_flag,
            "Sodium_over_2300_flag": sodium_over_2300_flag,
            "Sodium_over_1500_flag": sodium_over_1500_flag,
            "SatFat_over_10pct_flag": satfat_over_10pct_flag,
            "HDL_low_flag": hdl_low_flag,
            "TC_high_flag": tc_high_flag,
            "Comorbidity": comorbidity,

            "HTN_severity_score": htn_severity,
            "DM_severity_score": dm_severity,
            "Obesity_severity_score": obesity_severity,
            "Lifestyle_risk_score": lifestyle_risk,
            "Cardiometabolic_severity_score": cardiometabolic_score,
            "Severity_band": _severity_band(cardiometabolic_score),
            "data_completeness_score": data_completeness_score,

            "profile_age_group": _profile_age_group(age),
            "profile_cardiometabolic_risk": _profile_cardiometabolic_risk(cardiometabolic_score),
            "profile_glycaemic_status": _profile_glycaemic_status(diabetes, prediabetes, hba1c),
            "profile_bp_status": _profile_bp_status(hypertension, crisis_from_bp),
            "profile_weight_status": _profile_weight_status(bmi),
            "profile_version": "combined_patient_clinician_v1",
            "profile_data_completeness_score": data_completeness_score,
            "profile_current_smoker_flag": int(current_smoker),
            "profile_inactive_flag": inactive_signal_flag,
            "profile_high_alcohol_risk_flag": int(alcohol_over_guideline),
            "profile_poor_diet_signal_flag": int(
                sodium_over_2300_flag or satfat_over_10pct_flag or (fibre is not None and fibre < 25)
            ),

            "_record_count": len(self.records),
            "_clinician_record_count": len(self.clinician_records),
            "_patient_record_count": len(self.patient_records),
            "_latest_record_source_type": "clinician" if self.latest_clinician_record else ("patient" if self.latest_patient_record else None),
            "_latest_record_id": self.latest_record.id if self.latest_record else None,
            "_latest_recorded_at": self.latest_record.recorded_at.isoformat() if self.latest_record else None,
        }

        return payload

def build_current_health_feature_payload(user: User) -> dict[str, Any]:
    """
    Public service used by the live recommendation pipeline.

    Keeps external callers stable while the implementation supports:
    - clinician-entered dataset-aligned records
    - patient/family-entered records
    - profile fallback
    """
    return CurrentHealthFeatureBuilder(user).build()


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
