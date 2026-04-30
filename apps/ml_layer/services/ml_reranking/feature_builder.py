import pandas as pd

from apps.collaborative_layer.services.resolver import CollaborativeResolver
from apps.collaborative_layer.services.constants import COLLAB_DEFAULTS
from apps.collaborative_layer.services.snapshot import load_collaborative_snapshot
import json
from apps.patient_profile_layer.services.snapshot import load_patient_profiles_snapshot

COLLAB_FEATURE_COLUMNS = [
    "collab_available_flag",
    "collab_neighbor_count",
    "collab_support_count",
    "collab_support_rate",
    "collab_mean_rule_score_norm",
    "collab_mean_rank_pre_ml",
    "collab_family_support_rate",
    "collab_weighted_support_rate",
    "collab_weighted_mean_rule_score_norm",
    "collab_weighted_mean_rank_pre_ml",
    "collab_mean_supporting_similarity",
]

PATIENT_PROFILE_COLUMNS = [
    "participant_key",
    "age_group",
    "cardiometabolic_risk",
    "glycaemic_status",
    "bp_status",
    "weight_status",
    "lifestyle_flags",
    "data_completeness_score",
    "profile_version",
]



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
    """

    merged_df = ranked_df.merge(
        safety_context_df,
        on="ParticipantKey",
        how="left",
        validate="m:1",
    )

    # ---- Participant-level features ----
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

        merged_df = append_patient_profile_features(merged_df)
        merged_df = append_collaborative_features(merged_df)
        
        if "bundle_family" in merged_df.columns:
            merged_df = merged_df.sort_values(
                by=["ParticipantKey", "total_score", "confidence_score", "priority", "action_code"],
                ascending=[True, False, False, False, True],
                kind="stable",
            ).reset_index(drop=True)

            merged_df["bundle_rank_within_participant"] = (
                merged_df.groupby(["ParticipantKey", "bundle_family"]).cumcount() + 1
            )
        else:
            merged_df["bundle_rank_within_participant"] = 1

    # ---- Interaction features ----
    merged_df = add_interaction_features(merged_df)

    return merged_df


# -----------------------------------------------------
# Interaction features (rules × context × bundle)
# -----------------------------------------------------

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    bundle_family = (
        out["bundle_family"].astype(str)
        if "bundle_family" in out.columns
        else pd.Series("", index=out.index)
    )

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
        out["short_followup_flag"] = (
            pd.to_numeric(out["followup_horizon_days"], errors="coerce").fillna(365) <= 90
        ).astype(int)

    if "persistence_risk_proxy" in out.columns:
        out["high_persistence_risk_flag"] = (
            pd.to_numeric(out["persistence_risk_proxy"], errors="coerce").fillna(0) >= 2
        ).astype(int)

    if "longitudinal_ready_flag" in out.columns:
        out["longitudinal_ready_x_monitoring_bundle"] = (
            out["longitudinal_ready_flag"].fillna(0).astype(int) * out["is_monitoring_bundle"]
        )

    if "followup_horizon_days" in out.columns:
        horizon = pd.to_numeric(out["followup_horizon_days"], errors="coerce").fillna(365.0)
        out["followup_horizon_x_monitoring_bundle"] = horizon * out["is_monitoring_bundle"]

    return out

def _parse_lifestyle_flags(raw_value) -> set[str]:
    if raw_value is None or pd.isna(raw_value):
        return set()

    if isinstance(raw_value, list):
        return {str(v) for v in raw_value if str(v).strip()}

    if isinstance(raw_value, str):
        raw_value = raw_value.strip()
        if not raw_value:
            return set()
        try:
            parsed = json.loads(raw_value)
            if isinstance(parsed, list):
                return {str(v) for v in parsed if str(v).strip()}
        except json.JSONDecodeError:
            return {v for v in raw_value.split("|") if v}

    return set()


def append_patient_profile_features(features_df: pd.DataFrame) -> pd.DataFrame:
    profile_df = load_patient_profiles_snapshot(columns=PATIENT_PROFILE_COLUMNS)

    if profile_df.empty:
        raise RuntimeError(
            "Patient profile snapshot missing or empty. "
            "Run: python manage.py build_patient_profiles"
        )

    profile_df = profile_df.rename(
        columns={
            "participant_key": "ParticipantKey",
            "age_group": "profile_age_group",
            "cardiometabolic_risk": "profile_cardiometabolic_risk",
            "glycaemic_status": "profile_glycaemic_status",
            "bp_status": "profile_bp_status",
            "weight_status": "profile_weight_status",
            "lifestyle_flags": "profile_lifestyle_flags",
            "data_completeness_score": "profile_data_completeness_score",
            "profile_version": "profile_version",
        }
    )

    profile_df["ParticipantKey"] = profile_df["ParticipantKey"].astype(str)

    profile_df = profile_df.drop_duplicates(
        subset=["ParticipantKey"],
        keep="last",
    )

    out = features_df.copy()
    out["ParticipantKey"] = out["ParticipantKey"].astype(str)
    
    existing_profile_cols = [
        c for c in out.columns
        if c.startswith("profile_")
    ]

    if existing_profile_cols:
        out = out.drop(columns=existing_profile_cols)

    out = out.merge(
        profile_df,
        on="ParticipantKey",
        how="left",
        validate="many_to_one",
    )

    for col in [
        "profile_age_group",
        "profile_cardiometabolic_risk",
        "profile_glycaemic_status",
        "profile_bp_status",
        "profile_weight_status",
        "profile_version",
    ]:
        out[col] = out[col].fillna("unknown")

    out["profile_lifestyle_flags"] = out["profile_lifestyle_flags"].fillna("[]")

    out["profile_data_completeness_score"] = (
        pd.to_numeric(out["profile_data_completeness_score"], errors="coerce")
        .fillna(0.0)
        .astype("float32")
    )

    parsed_flags = out["profile_lifestyle_flags"].map(_parse_lifestyle_flags)

    out["profile_current_smoker_flag"] = parsed_flags.map(
        lambda flags: int("current_smoker" in flags)
    )

    out["profile_inactive_flag"] = parsed_flags.map(
        lambda flags: int("inactive" in flags)
    )

    out["profile_high_alcohol_risk_flag"] = parsed_flags.map(
        lambda flags: int("high_alcohol_risk" in flags)
    )

    out["profile_poor_diet_signal_flag"] = parsed_flags.map(
        lambda flags: int("poor_diet_signal" in flags)
    )

    return out

def append_collaborative_features(features_df: pd.DataFrame) -> pd.DataFrame:
    snapshot_df = load_collaborative_snapshot()

    if snapshot_df.empty:
        raise RuntimeError(
            "Collaborative snapshot missing or empty. "
            "Run: python manage.py build_collab_snapshot"
        )

    merge_keys = ["ParticipantKey", "action_code"]

    keep_cols = merge_keys + [
        c for c in COLLAB_FEATURE_COLUMNS if c in snapshot_df.columns
    ]

    before_cols = set(features_df.columns)

    out = features_df.merge(
        snapshot_df[keep_cols],
        on=merge_keys,
        how="left",
        validate="many_to_one",
    )

    added_cols = [c for c in out.columns if c not in before_cols]

    for col in added_cols:
        if col.startswith("collab_"):
            out[col] = out[col].fillna(0)

    return out
