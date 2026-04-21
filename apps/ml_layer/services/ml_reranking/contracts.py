import pandas as pd

from apps.common.clinical_safety.severity import is_valid_severity


REQUIRED_SAFETY_CONTEXT_COLUMNS = [
    "ParticipantKey",
    "feature_tier",
    "data_completeness_score",
    "triage_mode",
    "confidence_ceiling_global",
]

REQUIRED_RANKED_COLUMNS = [
    "ParticipantKey",
    "action_code",
    "action_type",
    "severity",
    "priority",
    "tags",
    "evidence",
    "source_rule_ids",
    "bundle_family",
    "coldstart_note_codes",
    "coldstart_confidence_ceiling",
    "base_severity_score",
    "base_action_score",
    "evidence_strength",
    "domain_alignment",
    "missing_penalty",
    "contraindication_penalty",
    "total_score",
    "score_norm",
    "confidence_score",
    "confidence_label",
]

OPTIONAL_PARTICIPANT_FEATURE_COLUMNS = [
    "ParticipantKey",
    "RIDAGEYR",
    "RIAGENDR",
    "BMXBMI",
    "BMXWAIST",
    "LBXGH",
    "SBP_mean",
    "DBP_mean",
    "Any_MVPA_flag",
    "Current_smoker_flag",
    "DietDay1_reliable_flag",
    "DR1TSODI",
    "DR1TSUGR",
    "DR1TFIBE",
    "Alcohol_over_guideline_flag",
    "Hypertension",
    "Hypertensive_crisis_flag",
    "Diabetes",
    "Prediabetes",
    "Severe_hyperglycemia_flag",
    "Severe_obesity_flag",
    "Sedentary_high_flag",
    "Sodium_over_2300_flag",
    "Sodium_over_1500_flag",
    "SatFat_over_10pct_flag",
    "HDL_low_flag",
    "TC_high_flag",
    "Comorbidity",
    "HTN_severity_score",
    "DM_severity_score",
    "Obesity_severity_score",
    "Lifestyle_risk_score",
    "Cardiometabolic_severity_score",
    "Severity_band",
    "followup_urgency",
    "followup_horizon_days",
    "monitoring_need_level",
    "temporal_risk_tier",
    "persistence_risk_proxy",
    "repeat_measurement_available",
    "temporal_data_quality",
    "longitudinal_ready_flag",
]

ML_ADDED_COLUMNS = [
    "ml_score",
    "lambda_u",
    "ml_contrib",
    "final_score",
    "rerank_status",
    "fallback_reason",
    "ml_model_version",
    "ml_schema_version",
    "ml_calibration_version",
]

FINAL_RERANKED_COLUMNS = REQUIRED_RANKED_COLUMNS + ML_ADDED_COLUMNS


def validate_safety_context_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_SAFETY_CONTEXT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required safety_context columns: {missing}")


def validate_ranked_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_RANKED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required ranked_recommendations columns: {missing}")

    if "ParticipantKey" in df.columns and df["ParticipantKey"].isna().any():
        raise ValueError("ranked_recommendations contains null ParticipantKey values")

    if "action_code" in df.columns and df["action_code"].isna().any():
        raise ValueError("ranked_recommendations contains null action_code values")

    invalid = []
    if "severity" in df.columns:
        invalid = sorted({
            str(v) for v in df["severity"].dropna().astype(str).unique()
            if not is_valid_severity(v)
        })

    if invalid:
        raise ValueError(f"Invalid severity values in ranked_recommendations: {invalid}")

    allowed_confidence = {"low", "medium", "high"}
    if "confidence_label" in df.columns:
        invalid_conf = sorted({
            str(v) for v in df["confidence_label"].dropna().astype(str).unique()
            if str(v) not in allowed_confidence
        })
        if invalid_conf:
            raise ValueError(f"Invalid confidence_label values in ranked_recommendations: {invalid_conf}")

    for col in ["priority", "total_score", "score_norm", "confidence_score"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            if s.isna().any():
                raise ValueError(f"ranked_recommendations contains non-numeric or missing values in {col}")


def validate_participant_features_columns(df: pd.DataFrame) -> None:
    if "ParticipantKey" not in df.columns:
        raise ValueError("participant_features_df must contain 'ParticipantKey'")


def ensure_ml_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ML_ADDED_COLUMNS:
        if col not in out.columns:
            out[col] = None
    return out
