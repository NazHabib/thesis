from django.conf import settings
from django.db import models
from django.utils import timezone


class PatientProfile(models.Model):
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="patient_profile",
    )

    # Stable patient context
    date_of_birth = models.DateField(null=True, blank=True)
    sex = models.CharField(max_length=20, blank=True, default="")

    has_dm = models.BooleanField(default=False)
    has_htn = models.BooleanField(default=False)

    # Optional care targets
    target_glucose_min = models.FloatField(null=True, blank=True)
    target_glucose_max = models.FloatField(null=True, blank=True)
    target_bp_sys = models.IntegerField(null=True, blank=True)
    target_bp_dia = models.IntegerField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)


class PatientHealthRecord(models.Model):
    """
    Append-only patient/family-entered health record.

    This model stores user-enterable data that can be mapped to the engineered
    feature space used by the cold-start, ranking, and ML reranking layers.

    It deliberately excludes system-generated fields such as:
    - action_code
    - bundle_family
    - confidence_label
    - collab_* features
    - fusion scores
    - SHAP/XAI payloads
    """

    class Source(models.TextChoices):
        PATIENT = "patient", "Patient"
        FAMILY = "family", "Family member / caregiver"
        HOME_DEVICE = "home_device", "Home device"
        LAB_RESULT = "lab_result", "Lab result"
        CLINIC = "clinic", "Clinic visit"
        OTHER = "other", "Other"

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="health_records",
    )
    
    entered_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="entered_health_records",
        help_text="User who entered this health record. May be the patient, family member, clinician, or admin.",
    )

    recorded_at = models.DateField(default=timezone.localdate)
    source = models.CharField(max_length=30, choices=Source.choices, default=Source.PATIENT)
    label = models.CharField(max_length=120, blank=True, default="Health record")

    # Demographic/context override for this record, optional.
    # Stable values still live in PatientProfile.
    age_years = models.IntegerField(null=True, blank=True)  # maps to RIDAGEYR if provided
    sex_code = models.IntegerField(null=True, blank=True)  # maps to RIAGENDR: 1 male, 2 female if provided

    # Anthropometrics
    height_cm = models.FloatField(null=True, blank=True)  # source for BMI calculation
    weight_kg = models.FloatField(null=True, blank=True)  # source for BMI calculation
    bmi = models.FloatField(null=True, blank=True)  # maps to BMXBMI
    waist_cm = models.FloatField(null=True, blank=True)  # maps to BMXWAIST

    # Glycaemic marker
    hba1c = models.FloatField(null=True, blank=True)  # maps to LBXGH

    # Blood pressure
    systolic_bp = models.FloatField(null=True, blank=True)  # maps to SBP_mean
    diastolic_bp = models.FloatField(null=True, blank=True)  # maps to DBP_mean
    bp_readings_count = models.IntegerField(null=True, blank=True)  # maps to BP_n_used

    # Diet / nutrition
    diet_day_reliable = models.BooleanField(default=True)  # maps to DietDay1_reliable_flag
    calories_kcal_day = models.FloatField(null=True, blank=True)  # maps to DR1TKCAL
    sodium_mg_day = models.FloatField(null=True, blank=True)  # maps to DR1TSODI
    sugar_g_day = models.FloatField(null=True, blank=True)  # maps to DR1TSUGR
    fibre_g_day = models.FloatField(null=True, blank=True)  # maps to DR1TFIBE
    saturated_fat_g_day = models.FloatField(null=True, blank=True)  # source for SatFat_energy_pct approx
    alcohol_g_day = models.FloatField(null=True, blank=True)  # maps to DR1TALCO approx

    # Lifestyle
    mvpa_minutes_week = models.IntegerField(null=True, blank=True)  # source for Any_MVPA_flag
    sedentary_hours_day = models.FloatField(null=True, blank=True)  # source for Sedentary_high_flag
    current_smoker = models.BooleanField(default=False)  # maps to Current_smoker_flag
    alcohol_over_guideline = models.BooleanField(default=False)  # maps to Alcohol_over_guideline_flag

    # Lipids
    total_cholesterol_mg_dl = models.FloatField(null=True, blank=True)  # maps to TotalChol_mg_dL
    hdl_mg_dl = models.FloatField(null=True, blank=True)  # maps to HDL_mg_dL

    # Optional flags entered from known results/context.
    # These can be derived, but storing optional user/clinician-entered context is useful.
    known_prediabetes = models.BooleanField(default=False)
    known_severe_hyperglycemia = models.BooleanField(default=False)
    known_hypertensive_crisis = models.BooleanField(default=False)

    notes = models.TextField(blank=True, default="")

    # Optional audit/debug payload. Not shown to normal patient pages.
    raw_payload = models.JSONField(blank=True, default=dict)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-recorded_at", "-created_at"]

    def __str__(self):
        return f"{self.user} health record on {self.recorded_at}"

    def calculate_bmi_if_possible(self):
        if self.bmi:
            return self.bmi

        if not self.height_cm or not self.weight_kg:
            return None

        height_m = self.height_cm / 100
        if height_m <= 0:
            return None

        return round(self.weight_kg / (height_m * height_m), 1)

class ClinicianHealthRecord(models.Model):
    """
    Clinician-entered, dataset-aligned health record.

    Separate from PatientHealthRecord because this captures more detailed,
    clinician-reviewed / NHANES-style fields.
    """

    class Source(models.TextChoices):
        CLINIC = "clinic", "Clinic visit"
        LAB_RESULT = "lab_result", "Lab result"
        CLINICIAN_REVIEW = "clinician_review", "Clinician review"
        IMPORTED_RECORD = "imported_record", "Imported clinical record"
        OTHER = "other", "Other"

    patient = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="clinician_health_records",
    )

    clinician = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="clinician_entered_health_records",
    )

    recorded_at = models.DateField(default=timezone.localdate)
    source = models.CharField(max_length=40, choices=Source.choices, default=Source.CLINIC)
    label = models.CharField(max_length=140, blank=True, default="Clinician-entered health record")

    age_years = models.IntegerField(null=True, blank=True)
    sex_code = models.IntegerField(null=True, blank=True)

    bpxsy1 = models.FloatField(null=True, blank=True)
    bpxsy2 = models.FloatField(null=True, blank=True)
    bpxsy3 = models.FloatField(null=True, blank=True)
    bpxsy4 = models.FloatField(null=True, blank=True)

    bpxdi1 = models.FloatField(null=True, blank=True)
    bpxdi2 = models.FloatField(null=True, blank=True)
    bpxdi3 = models.FloatField(null=True, blank=True)
    bpxdi4 = models.FloatField(null=True, blank=True)

    pulse_bpm = models.FloatField(null=True, blank=True)

    systolic_bp = models.FloatField(null=True, blank=True)
    diastolic_bp = models.FloatField(null=True, blank=True)
    bp_readings_count = models.IntegerField(null=True, blank=True)

    height_cm = models.FloatField(null=True, blank=True)
    weight_kg = models.FloatField(null=True, blank=True)
    bmi = models.FloatField(null=True, blank=True)
    waist_cm = models.FloatField(null=True, blank=True)
    hip_cm = models.FloatField(null=True, blank=True)

    hba1c = models.FloatField(null=True, blank=True)

    total_cholesterol_mg_dl = models.FloatField(null=True, blank=True)
    hdl_mg_dl = models.FloatField(null=True, blank=True)

    diet_day_reliable = models.BooleanField(default=True)
    calories_kcal_day = models.FloatField(null=True, blank=True)
    protein_g_day = models.FloatField(null=True, blank=True)
    carbohydrate_g_day = models.FloatField(null=True, blank=True)
    total_fat_g_day = models.FloatField(null=True, blank=True)
    saturated_fat_g_day = models.FloatField(null=True, blank=True)
    sugar_g_day = models.FloatField(null=True, blank=True)
    fibre_g_day = models.FloatField(null=True, blank=True)
    sodium_mg_day = models.FloatField(null=True, blank=True)
    potassium_mg_day = models.FloatField(null=True, blank=True)
    alcohol_g_day = models.FloatField(null=True, blank=True)

    smoking_status = models.CharField(
        max_length=30,
        blank=True,
        default="",
        choices=[
            ("", "Unknown / not recorded"),
            ("never", "Never smoker"),
            ("former", "Former smoker"),
            ("current", "Current smoker"),
        ],
    )
    current_smoker = models.BooleanField(default=False)
    mvpa_minutes_week = models.IntegerField(null=True, blank=True)
    vigorous_activity = models.BooleanField(default=False)
    moderate_activity = models.BooleanField(default=False)
    sedentary_hours_day = models.FloatField(null=True, blank=True)
    alcohol_over_guideline = models.BooleanField(default=False)

    known_prediabetes = models.BooleanField(default=False)
    known_severe_hyperglycemia = models.BooleanField(default=False)
    known_hypertensive_crisis = models.BooleanField(default=False)

    notes = models.TextField(blank=True, default="")
    raw_payload = models.JSONField(blank=True, default=dict)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-recorded_at", "-created_at"]

    def __str__(self):
        return f"{self.patient} clinician health record on {self.recorded_at}"

    def calculate_bmi_if_possible(self):
        if self.bmi:
            return self.bmi

        if not self.height_cm or not self.weight_kg:
            return None

        height_m = self.height_cm / 100
        if height_m <= 0:
            return None

        return round(self.weight_kg / (height_m * height_m), 1)

    def mean_sbp_from_readings(self):
        values = [self.bpxsy1, self.bpxsy2, self.bpxsy3, self.bpxsy4]
        nums = [float(v) for v in values if v not in [None, ""]]
        return round(sum(nums) / len(nums), 2) if nums else None

    def mean_dbp_from_readings(self):
        values = [self.bpxdi1, self.bpxdi2, self.bpxdi3, self.bpxdi4]
        nums = [float(v) for v in values if v not in [None, ""]]
        return round(sum(nums) / len(nums), 2) if nums else None

    def bp_reading_count_from_readings(self):
        sbp_count = sum(1 for v in [self.bpxsy1, self.bpxsy2, self.bpxsy3, self.bpxsy4] if v not in [None, ""])
        dbp_count = sum(1 for v in [self.bpxdi1, self.bpxdi2, self.bpxdi3, self.bpxdi4] if v not in [None, ""])
        return min(sbp_count, dbp_count) or None

class ClinicianPatientLink(models.Model):
    clinician = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="assigned_patients",
    )
    patient = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="assigned_clinicians",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("clinician", "patient")

    def __str__(self):
        return f"{self.clinician.username} -> {self.patient.username}"
