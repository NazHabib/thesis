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

PATIENT_SAFE_EXPLAINABILITY_FIELDS = [
    "ParticipantKey",
    "bundle_title",
    "bundle_family",
    "category",
    "action_codes",
    "action_type",
    "priority",
    "confidence_score",
    "confidence_label",
    "patient_facing_explanation",
    "patient_confidence_explanation",
    "patient_ml_xai_explanation",
    "safety_context_note",
    "followup_window_label",
    "followup_timing_text",
]

CLINICIAN_EXPLAINABILITY_FIELDS = [
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
    "bundle_score",
    "confidence_score",
    "confidence_label",
    "clinician_facing_explanation",
    "clinician_confidence_explanation",
    "clinician_ml_xai_explanation",
    "ml_rerank_note",
    "traceability_payload",
    "ml_xai_payload",
    "fusion_strategy",
    "fusion_version",
    "fusion_status",
]


def validate_final_bundle_columns(df):
    missing = [c for c in REQUIRED_FINAL_BUNDLE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required final recommendation columns: {missing}")


def validate_safety_context_columns(df):
    missing = [c for c in REQUIRED_SAFETY_CONTEXT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required safety context columns: {missing}")
