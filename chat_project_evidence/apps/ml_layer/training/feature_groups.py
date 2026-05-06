from __future__ import annotations

TEMPORAL_CATEGORICAL_COLUMNS = [
    "followup_urgency",
    "monitoring_need_level",
    "temporal_risk_tier",
    "temporal_data_quality",
]

TEMPORAL_NUMERIC_COLUMNS = [
    "followup_horizon_days",
    "persistence_risk_proxy",
    "repeat_measurement_available",
    "longitudinal_ready_flag",
]

TEMPORAL_DERIVED_COLUMNS = [
    "short_followup_flag",
    "high_persistence_risk_flag",
    "longitudinal_ready_x_monitoring_bundle",
    "followup_horizon_x_monitoring_bundle",
]

COLLABORATIVE_FEATURE_COLUMNS = [
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

PATIENT_PROFILE_CATEGORICAL_COLUMNS = [
    "profile_age_group",
    "profile_cardiometabolic_risk",
    "profile_glycaemic_status",
    "profile_bp_status",
    "profile_weight_status",
    "profile_version",
]

PATIENT_PROFILE_NUMERIC_COLUMNS = [
    "profile_data_completeness_score",
    "profile_current_smoker_flag",
    "profile_inactive_flag",
    "profile_high_alcohol_risk_flag",
    "profile_poor_diet_signal_flag",
]

PATIENT_PROFILE_FEATURE_COLUMNS = (
    PATIENT_PROFILE_CATEGORICAL_COLUMNS
    + PATIENT_PROFILE_NUMERIC_COLUMNS
)

ALL_TEMPORAL_FEATURE_COLUMNS = (
    TEMPORAL_CATEGORICAL_COLUMNS
    + TEMPORAL_NUMERIC_COLUMNS
    + TEMPORAL_DERIVED_COLUMNS
)

TEMPORAL_ONLY_STRUCTURAL_CATEGORICAL_COLUMNS = [
    "action_code",
    "action_type",
    "severity",
    "bundle_family",
    "confidence_label",
    "feature_tier",
    "confidence_ceiling_global",
    "coldstart_confidence_ceiling",
    "Severity_band",
]

TEMPORAL_ONLY_STRUCTURAL_NUMERIC_COLUMNS = [
    "data_completeness_score",
]


def remove_temporal_columns(columns: list[str]) -> list[str]:
    temporal = set(ALL_TEMPORAL_FEATURE_COLUMNS)
    return [c for c in columns if c not in temporal]


def remove_collaborative_columns(columns: list[str]) -> list[str]:
    collab = set(COLLABORATIVE_FEATURE_COLUMNS)
    return [c for c in columns if c not in collab]


def keep_temporal_only_columns(
    categorical_columns: list[str],
    numeric_columns: list[str],
) -> tuple[list[str], list[str]]:
    allowed_categorical = set(
        TEMPORAL_CATEGORICAL_COLUMNS
        + TEMPORAL_ONLY_STRUCTURAL_CATEGORICAL_COLUMNS
    )
    allowed_numeric = set(
        TEMPORAL_NUMERIC_COLUMNS
        + TEMPORAL_DERIVED_COLUMNS
        + TEMPORAL_ONLY_STRUCTURAL_NUMERIC_COLUMNS
    )

    cat_out = [c for c in categorical_columns if c in allowed_categorical]
    num_out = [c for c in numeric_columns if c in allowed_numeric]

    return cat_out, num_out


def temporal_feature_count(columns: list[str]) -> int:
    temporal = set(ALL_TEMPORAL_FEATURE_COLUMNS)
    return sum(1 for c in columns if c in temporal)


def collaborative_feature_count(columns: list[str]) -> int:
    collab = set(COLLABORATIVE_FEATURE_COLUMNS)
    return sum(1 for c in columns if c in collab)


def temporal_feature_subset(columns: list[str]) -> list[str]:
    temporal = set(ALL_TEMPORAL_FEATURE_COLUMNS)
    return [c for c in columns if c in temporal]


def collaborative_feature_subset(columns: list[str]) -> list[str]:
    collab = set(COLLABORATIVE_FEATURE_COLUMNS)
    return [c for c in columns if c in collab]
    
def remove_patient_profile_columns(columns: list[str]) -> list[str]:
    profile = set(PATIENT_PROFILE_FEATURE_COLUMNS)
    return [c for c in columns if c not in profile]


def patient_profile_feature_count(columns: list[str]) -> int:
    profile = set(PATIENT_PROFILE_FEATURE_COLUMNS)
    return sum(1 for c in columns if c in profile)


def patient_profile_feature_subset(columns: list[str]) -> list[str]:
    profile = set(PATIENT_PROFILE_FEATURE_COLUMNS)
    return [c for c in columns if c in profile]
