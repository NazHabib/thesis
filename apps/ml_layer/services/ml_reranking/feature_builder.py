import pandas as pd


PARTICIPANT_FEATURE_COLUMNS = [
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


def build_ml_feature_table(
    safety_context_df: pd.DataFrame,
    ranked_df: pd.DataFrame,
    participant_features_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Merge the pre-ML inputs into one participant-candidate table.

    This does not perform ML preprocessing yet.
    """
    merged_df = ranked_df.merge(
        safety_context_df,
        on="ParticipantKey",
        how="left",
        validate="m:1",
    )

    if participant_features_df is not None and not participant_features_df.empty:
        available_feature_cols = [
    	    c for c in PARTICIPANT_FEATURE_COLUMNS
    	    if c in participant_features_df.columns
    	    and (c == "ParticipantKey" or c not in merged_df.columns)
	]
        participant_df = (
            participant_features_df[available_feature_cols]
            .drop_duplicates(subset=["ParticipantKey"])
            .copy()
        )

        merged_df = merged_df.merge(
            participant_df,
            on="ParticipantKey",
            how="left",
            validate="m:1",
        )

    merged_df = add_interaction_features(merged_df)

    return merged_df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    available_feature_cols = [
        c for c in PARTICIPANT_FEATURE_COLUMNS
        if c in participant_features_df.columns
        and (c == "ParticipantKey" or c not in merged_df.columns)
    ]

    out["is_diet_bundle"] = bundle_family.eq("DietPattern").astype(int)
    out["is_movement_bundle"] = bundle_family.eq("Movement").astype(int)
    out["is_weight_bundle"] = bundle_family.eq("Weight").astype(int)
    out["is_smoking_bundle"] = bundle_family.eq("Smoking").astype(int)
    out["is_monitoring_bundle"] = bundle_family.eq("Monitoring").astype(int)

    if "LBXGH" in out.columns:
        out["hba1c_high_flag"] = (out["LBXGH"].fillna(0.0) >= 7.0).astype(int)
        out["hba1c_high_x_diet_bundle"] = out["hba1c_high_flag"] * out["is_diet_bundle"]
        out["hba1c_high_x_monitoring_bundle"] = out["hba1c_high_flag"] * out["is_monitoring_bundle"]

    if "SBP_mean" in out.columns:
        out["sbp_high_flag"] = (out["SBP_mean"].fillna(0.0) >= 130.0).astype(int)
        out["sbp_high_x_movement_bundle"] = out["sbp_high_flag"] * out["is_movement_bundle"]
        out["sbp_high_x_monitoring_bundle"] = out["sbp_high_flag"] * out["is_monitoring_bundle"]

    if "BMXBMI" in out.columns:
        out["obesity_flag_proxy"] = (out["BMXBMI"].fillna(0.0) >= 30.0).astype(int)
        out["bmi_x_weight_bundle"] = out["BMXBMI"].fillna(0.0) * out["is_weight_bundle"]

    if "Current_smoker_flag" in out.columns:
        out["smoker_x_smoking_bundle"] = out["Current_smoker_flag"].fillna(0.0) * out["is_smoking_bundle"]

    if "Any_MVPA_flag" in out.columns:
        out["inactive_flag_proxy"] = (out["Any_MVPA_flag"].fillna(0.0) == 0.0).astype(int)
        out["inactive_x_movement_bundle"] = out["inactive_flag_proxy"] * out["is_movement_bundle"]

    if "Sodium_over_2300_flag" in out.columns:
        out["high_sodium_x_diet_bundle"] = out["Sodium_over_2300_flag"].fillna(0.0) * out["is_diet_bundle"]
        
    if "followup_horizon_days" in out.columns:
        out["short_followup_flag"] = (pd.to_numeric(out["followup_horizon_days"], errors="coerce").fillna(365) <= 90).astype(int)

    if "persistence_risk_proxy" in out.columns:
        out["high_persistence_risk_flag"] = (pd.to_numeric(out["persistence_risk_proxy"], errors="coerce").fillna(0) >= 2).astype(int)

    if "longitudinal_ready_flag" in out.columns:
        out["longitudinal_ready_x_monitoring_bundle"] = out["longitudinal_ready_flag"].fillna(0).astype(int) * out["is_monitoring_bundle"]

    if "followup_horizon_days" in out.columns and "is_monitoring_bundle" in out.columns:
        horizon = pd.to_numeric(out["followup_horizon_days"], errors="coerce").fillna(365.0)
        out["followup_horizon_x_monitoring_bundle"] = horizon * out["is_monitoring_bundle"]

    return out
