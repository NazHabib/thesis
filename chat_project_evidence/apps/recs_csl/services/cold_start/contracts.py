REQUIRED_CFM_COLUMNS = [
    "ParticipantKey",
    "Cycle",
    "SBP_mean",
    "DBP_mean",
    "BP_n_used",
    "LBXGH",
    "BMXBMI",
    "DietDay1_reliable_flag",
    "DR1TSODI",
    "Current_smoker_flag",
    "Any_MVPA_flag",
    "Hypertensive_crisis_flag",
    "Severe_hyperglycemia_flag",
]

REQUIRED_CANDIDATE_FIELDS = [
    "action_code",
    "action_type",
    "severity",
    "priority",
    "tags",
    "evidence",
    "source_rule_ids",
]


def validate_cfm_columns(df):
    missing = [c for c in REQUIRED_CFM_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required CFM columns: {missing}")


def validate_candidates_list(candidates):
    if not isinstance(candidates, list):
        raise ValueError("Candidates must be a list.")

    for i, c in enumerate(candidates):
        missing = [field for field in REQUIRED_CANDIDATE_FIELDS if field not in c]
        if missing:
            raise ValueError(f"Candidate index {i} missing fields: {missing}")
