FEATURE_LABELS = {
    # Core clinical biomarkers
    "num__LBXGH": "HbA1c",
    "num__SBP_mean": "Mean systolic blood pressure",
    "num__DBP_mean": "Mean diastolic blood pressure",
    "num__BMXBMI": "Body mass index",
    "num__BMXWAIST": "Waist circumference",
    "num__DR1TSODI": "Day-1 sodium intake",
    "num__DR1TFIBE": "Day-1 fibre intake",
    "num__DR1TSUGR": "Day-1 total sugar intake",
    "num__TotalChol_mg_dL": "Total cholesterol",
    "num__HDL_mg_dL": "HDL cholesterol",

    # Risk / status flags
    "num__Diabetes": "Diabetes flag",
    "num__Prediabetes": "Prediabetes flag",
    "num__Hypertension": "Hypertension flag",
    "num__Hypertensive_crisis_flag": "Hypertensive crisis flag",
    "num__Severe_hyperglycemia_flag": "Severe hyperglycaemia flag",
    "num__Current_smoker_flag": "Current smoker flag",
    "num__Any_MVPA_flag": "Any moderate/vigorous activity flag",
    "num__Sedentary_high_flag": "High sedentary time flag",
    "num__Sodium_over_2300_flag": "Sodium above 2300 mg flag",
    "num__Sodium_over_1500_flag": "Sodium above 1500 mg flag",
    "num__SatFat_over_10pct_flag": "Saturated fat above 10% energy flag",
    "num__Alcohol_over_guideline_flag": "Alcohol above guideline flag",
    "num__HDL_low_flag": "Low HDL flag",
    "num__TC_high_flag": "High total cholesterol flag",
    "num__Severe_obesity_flag": "Severe obesity flag",

    # Severity / engineered clinical context
    "num__HTN_severity_score": "Hypertension severity score",
    "num__DM_severity_score": "Diabetes severity score",
    "num__Obesity_severity_score": "Obesity severity score",
    "num__Lifestyle_risk_score": "Lifestyle risk score",
    "num__Cardiometabolic_severity_score": "Cardiometabolic severity score",
    "cat__Severity_band_Low": "Low severity band",
    "cat__Severity_band_Moderate": "Moderate severity band",
    "cat__Severity_band_High": "High severity band",

    # Upstream ranking / safety context
    "num__total_score": "Rule-based total score",
    "num__score_norm": "Normalized rule score",
    "num__confidence_score": "Upstream confidence score",
    "cat__confidence_label_high": "High upstream confidence",
    "cat__confidence_label_medium": "Medium upstream confidence",
    "cat__confidence_label_low": "Low upstream confidence",
    "num__priority": "Recommendation priority",
    "num__domain_alignment": "Domain alignment score",
    "num__evidence_strength": "Evidence strength score",
    "num__base_action_score": "Base action score",
    "num__base_severity_score": "Base severity score",
    "num__missing_penalty": "Missing-data penalty",
    "num__contraindication_penalty": "Contraindication penalty",
    "num__data_completeness_score": "Data completeness score",

    # Candidate/bundle context
    "cat__bundle_family_DietPattern": "Diet pattern bundle family",
    "cat__bundle_family_Movement": "Movement bundle family",
    "cat__bundle_family_Monitoring": "Monitoring bundle family",
    "cat__bundle_family_Smoking": "Smoking bundle family",
    "cat__bundle_family_Lipids": "Lipids bundle family",
    "cat__bundle_family_Weight": "Weight bundle family",
    "cat__bundle_family_Alcohol": "Alcohol bundle family",
    "cat__bundle_family_IntegratedCardiometabolic": "Integrated cardiometabolic bundle family",

    "cat__action_type_recommend": "Recommendation action type",
    "cat__action_type_inform": "Informational action type",
    "cat__action_type_escalate": "Escalation action type",

    "cat__severity_low": "Low candidate severity",
    "cat__severity_moderate": "Moderate candidate severity",
    "cat__severity_high": "High candidate severity",
    "cat__severity_critical": "Critical candidate severity",
    
    "num__profile_data_completeness_score": "Patient profile completeness score",
    "num__profile_current_smoker_flag": "Profile-derived current smoker flag",
    "num__profile_inactive_flag": "Profile-derived inactivity flag",
    "num__profile_high_alcohol_risk_flag": "Profile-derived high alcohol risk flag",
    "num__profile_poor_diet_signal_flag": "Profile-derived poor diet signal flag",

    "cat__profile_age_group_under_18": "Profile age group: under 18",
    "cat__profile_age_group_18_39": "Profile age group: 18–39",
    "cat__profile_age_group_40_64": "Profile age group: 40–64",
    "cat__profile_age_group_65_plus": "Profile age group: 65+",

    "cat__profile_cardiometabolic_risk_low": "Profile cardiometabolic risk: low",
    "cat__profile_cardiometabolic_risk_moderate": "Profile cardiometabolic risk: moderate",
    "cat__profile_cardiometabolic_risk_high": "Profile cardiometabolic risk: high",
    "cat__profile_cardiometabolic_risk_very_high": "Profile cardiometabolic risk: very high",

    "cat__profile_glycaemic_status_normal": "Profile glycaemic status: normal",
    "cat__profile_glycaemic_status_prediabetes_range": "Profile glycaemic status: prediabetes range",
    "cat__profile_glycaemic_status_diabetes_range": "Profile glycaemic status: diabetes range",
    "cat__profile_glycaemic_status_at_target_known_diabetes": "Profile glycaemic status: known diabetes at target",
    "cat__profile_glycaemic_status_above_target_known_diabetes": "Profile glycaemic status: known diabetes above target",

    "cat__profile_bp_status_normal": "Profile blood pressure status: normal",
    "cat__profile_bp_status_elevated": "Profile blood pressure status: elevated",
    "cat__profile_bp_status_stage_1_hypertension": "Profile blood pressure status: stage 1 hypertension",
    "cat__profile_bp_status_stage_2_hypertension": "Profile blood pressure status: stage 2 hypertension",
    "cat__profile_bp_status_severe_hypertension": "Profile blood pressure status: severe hypertension",

    "cat__profile_weight_status_underweight": "Profile weight status: underweight",
    "cat__profile_weight_status_normal": "Profile weight status: normal",
    "cat__profile_weight_status_overweight": "Profile weight status: overweight",
    "cat__profile_weight_status_obesity": "Profile weight status: obesity",
    
    # Patient profile features
    "cat__profile_age_group_18_39": "Age group 18–39",
    "cat__profile_age_group_40_64": "Age group 40–64",
    "cat__profile_age_group_65_plus": "Age group 65+",
    "cat__profile_cardiometabolic_risk_low": "Low cardiometabolic risk",
    "cat__profile_cardiometabolic_risk_moderate": "Moderate cardiometabolic risk",
    "cat__profile_cardiometabolic_risk_high": "High cardiometabolic risk",
    "cat__profile_cardiometabolic_risk_very_high": "Very high cardiometabolic risk",
    "cat__profile_glycaemic_status_normal": "Normal glycaemic status",
    "cat__profile_glycaemic_status_prediabetes_range": "Prediabetes range",
    "cat__profile_glycaemic_status_above_target_known_diabetes": "Diabetes above target",
    "cat__profile_bp_status_normal": "Normal blood pressure",
    "cat__profile_bp_status_stage_1_hypertension": "Stage 1 hypertension",
    "cat__profile_bp_status_stage_2_hypertension": "Stage 2 hypertension",
    "cat__profile_weight_status_normal": "Normal weight",
    "cat__profile_weight_status_overweight": "Overweight",
    "cat__profile_weight_status_obesity": "Obesity",

    "num__profile_data_completeness_score": "Profile data completeness",
    "num__profile_current_smoker_flag": "Profile smoker flag",
    "num__profile_inactive_flag": "Profile inactivity flag",
    "num__profile_high_alcohol_risk_flag": "Profile alcohol risk flag",
    "num__profile_poor_diet_signal_flag": "Profile poor diet signal",
}


META_RANKING_FEATURES = {
    "num__total_score",
    "num__score_norm",
    "num__confidence_score",
    "cat__confidence_label_high",
    "cat__confidence_label_medium",
    "cat__confidence_label_low",
    "num__priority",
    "num__domain_alignment",
    "num__evidence_strength",
    "num__base_action_score",
    "num__base_severity_score",
    "num__missing_penalty",
    "num__contraindication_penalty",
    "num__data_completeness_score",
}

SAFETY_CONTEXT_FEATURES = {
    "num__Hypertensive_crisis_flag",
    "num__Severe_hyperglycemia_flag",
    "cat__Severity_band_Low",
    "cat__Severity_band_Moderate",
    "cat__Severity_band_High",
}

CANDIDATE_CONTEXT_PREFIXES = (
    "cat__bundle_family_",
    "cat__action_type_",
    "cat__severity_",
)

LIFESTYLE_FEATURES = {
    "num__DR1TSODI",
    "num__DR1TFIBE",
    "num__DR1TSUGR",
    "num__Current_smoker_flag",
    "num__Any_MVPA_flag",
    "num__Sedentary_high_flag",
    "num__Sodium_over_2300_flag",
    "num__Sodium_over_1500_flag",
    "num__SatFat_over_10pct_flag",
    "num__Alcohol_over_guideline_flag",
}

CLINICAL_FEATURES = {
    "num__LBXGH",
    "num__SBP_mean",
    "num__DBP_mean",
    "num__BMXBMI",
    "num__BMXWAIST",
    "num__TotalChol_mg_dL",
    "num__HDL_mg_dL",
    "num__Diabetes",
    "num__Prediabetes",
    "num__Hypertension",
    "num__Severe_obesity_flag",
    "num__HDL_low_flag",
    "num__TC_high_flag",
    "num__HTN_severity_score",
    "num__DM_severity_score",
    "num__Obesity_severity_score",
    "num__Lifestyle_risk_score",
    "num__Cardiometabolic_severity_score",
}

PATIENT_PROFILE_FEATURES = {
    "num__profile_data_completeness_score",
    "num__profile_current_smoker_flag",
    "num__profile_inactive_flag",
    "num__profile_high_alcohol_risk_flag",
    "num__profile_poor_diet_signal_flag",
}


def humanize_feature_name(raw_name: str) -> str:
    raw_name = str(raw_name)
    return FEATURE_LABELS.get(raw_name, raw_name)


def feature_group(raw_name: str) -> str:
    raw_name = str(raw_name)

    if raw_name in META_RANKING_FEATURES:
        return "meta_ranking"
    if raw_name in SAFETY_CONTEXT_FEATURES:
        return "safety_context"
    if raw_name in CLINICAL_FEATURES:
        return "clinical"
    if raw_name in LIFESTYLE_FEATURES:
        return "lifestyle"
    if raw_name.startswith(CANDIDATE_CONTEXT_PREFIXES):
        return "candidate_context"
    if raw_name.startswith("cat__profile_") or raw_name in PATIENT_PROFILE_FEATURES:
        return "patient_profile"
    return "other"


def is_clinical_only_feature(raw_name: str) -> bool:
    return feature_group(raw_name) in {
        "clinical",
        "lifestyle",
        "safety_context",
        "patient_profile",
    }
