REQUIRED_FINAL_BUNDLE_COLUMNS = [
    "ParticipantKey",
    "bundle_title",
    "bundle_family",
    "category",
    "action_codes",
    "action_type",
    "priority",
    "tags",
    "source_rule_ids",
    "evidence",
    "coldstart_note_codes",
    "bundle_score",
    "confidence_score",
    "confidence_label",
    "rationale",
    "explanation",
    "triage_mode",
    "triage_reason",
]

REQUIRED_SAFETY_CONTEXT_COLUMNS = [
    "ParticipantKey",
    "feature_tier",
    "triage_mode",
    "triage_reason",
    "confidence_ceiling_global",
]


def validate_final_bundle_columns(df):
    missing = [c for c in REQUIRED_FINAL_BUNDLE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required final recommendation columns: {missing}")


def validate_safety_context_columns(df):
    missing = [c for c in REQUIRED_SAFETY_CONTEXT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required safety context columns: {missing}")
