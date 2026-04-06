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
