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


def keep_temporal_only_columns(
    categorical_columns: list[str],
    numeric_columns: list[str],
) -> tuple[list[str], list[str]]:
    allowed_categorical = set(TEMPORAL_CATEGORICAL_COLUMNS + TEMPORAL_ONLY_STRUCTURAL_CATEGORICAL_COLUMNS)
    allowed_numeric = set(TEMPORAL_NUMERIC_COLUMNS + TEMPORAL_DERIVED_COLUMNS + TEMPORAL_ONLY_STRUCTURAL_NUMERIC_COLUMNS)

    cat_out = [c for c in categorical_columns if c in allowed_categorical]
    num_out = [c for c in numeric_columns if c in allowed_numeric]

    return cat_out, num_out


def temporal_feature_count(columns: list[str]) -> int:
    temporal = set(ALL_TEMPORAL_FEATURE_COLUMNS)
    return sum(1 for c in columns if c in temporal)


def temporal_feature_subset(columns: list[str]) -> list[str]:
    temporal = set(ALL_TEMPORAL_FEATURE_COLUMNS)
    return [c for c in columns if c in temporal]
