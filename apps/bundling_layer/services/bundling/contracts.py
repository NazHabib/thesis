REQUIRED_SAFETY_CONTEXT_COLUMNS = [
    "ParticipantKey",
    "triage_mode",
    "triage_reason",
    "feature_tier",
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
    "total_score",
    "confidence_score",
    "confidence_label",
]


def validate_safety_context_columns(df):
    missing = [c for c in REQUIRED_SAFETY_CONTEXT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required safety_context columns: {missing}")


def validate_ranked_columns(df):
    missing = [c for c in REQUIRED_RANKED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required ranked_recommendations columns: {missing}")
