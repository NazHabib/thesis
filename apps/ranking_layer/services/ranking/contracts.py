from apps.common.clinical_safety.severity import is_valid_severity


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


def validate_gated_candidate_columns(df):
    missing = [c for c in REQUIRED_GATED_CANDIDATE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required gated_candidates columns: {missing}")

    invalid = []
    if "severity" in df.columns:
        invalid = sorted({
            str(v) for v in df["severity"].dropna().astype(str).unique()
            if not is_valid_severity(v)
        })

    if invalid:
        raise ValueError(f"Invalid severity values in gated_candidates: {invalid}")
