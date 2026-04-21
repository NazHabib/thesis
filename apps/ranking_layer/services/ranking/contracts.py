import pandas as pd
from apps.common.clinical_safety.severity import is_valid_severity


VALID_ACTION_TYPES = {"recommend", "inform", "forbid", "escalate"}


REQUIRED_SAFETY_CONTEXT_COLUMNS = [
    "ParticipantKey",
    "feature_tier",
    "data_completeness_score",
    "triage_mode",
    "confidence_ceiling_global",
]

REQUIRED_GATED_CANDIDATE_COLUMNS = [
    "ParticipantKey",
    "action_code",
    "action_type",
    "severity",
    "priority",
    "tags",
    "evidence",
    "source_rule_ids",
    "coldstart_confidence_ceiling",
]


def validate_safety_context_columns(df):
    missing = [c for c in REQUIRED_SAFETY_CONTEXT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required safety_context columns: {missing}")


def _ensure_list_column(df, col):
    if col in df.columns:
        bad = df[col].apply(lambda x: not isinstance(x, list))
        if bad.any():
            raise ValueError(f"{col} must contain lists")


def validate_gated_candidate_columns(df):
    missing = [c for c in REQUIRED_GATED_CANDIDATE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required gated_candidates columns: {missing}")

    # Severity validation
    invalid_severity = sorted({
        str(v) for v in df["severity"].dropna().astype(str).unique()
        if not is_valid_severity(v)
    })
    if invalid_severity:
        raise ValueError(f"Invalid severity values in gated_candidates: {invalid_severity}")

    # Action type validation
    invalid_action_types = sorted({
        str(v) for v in df["action_type"].dropna().astype(str).unique()
        if v not in VALID_ACTION_TYPES
    })
    if invalid_action_types:
        raise ValueError(f"Invalid action_type values in gated_candidates: {invalid_action_types}")

    # Priority validation
    if not pd.api.types.is_numeric_dtype(df["priority"]):
        raise ValueError("priority column must be numeric")

    if df["priority"].isna().any():
        raise ValueError("priority column contains missing values")

    # Structural fields validation
    _ensure_list_column(df, "tags")
    _ensure_list_column(df, "evidence")
    _ensure_list_column(df, "source_rule_ids")
