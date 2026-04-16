import pandas as pd

from .config import (
    SEVERITY_SCORE_MAP,
    EVIDENCE_BONUS_PER_PASSED,
    CONTRAINDICATION_UNKNOWN_PENALTY,
    MISSING_DATA_PENALTY,
)


SEVERITY_NORMALIZATION_MAP = {
    "info": "informational",
    "informational": "informational",
    "low": "low",
    "medium": "moderate",
    "moderate": "moderate",
    "high": "high",
    "critical": "critical",
}


def normalize_severity(value) -> str:
    return SEVERITY_NORMALIZATION_MAP.get(str(value).strip().lower(), str(value).strip().lower())


def count_passed_evidence(evidence_list) -> int:
    """
    Count unique passed evidence items to avoid inflated scores from duplicated
    evidence rows in the rules output.
    """
    if not isinstance(evidence_list, list):
        return 0

    seen = set()
    for e in evidence_list:
        if isinstance(e, dict) and e.get("passed") is True:
            key = (
                e.get("feature"),
                e.get("operator"),
                str(e.get("threshold")),
            )
            seen.add(key)

    return len(seen)


def compute_domain_alignment(row) -> int:
    """
    Heuristic alignment between candidate tags and participant clinical context.
    Keep this readable and deterministic.
    """
    tags = row["tags"] if isinstance(row["tags"], list) else []
    tags_set = {str(t).lower() for t in tags}

    score = 0

    if {"acute", "safety"} & tags_set and bool(row.get("triage_mode", False)):
        score += 15

    if {"diabetes", "a1c"} & tags_set and bool(row.get("a1c_core", False)):
        score += 8

    if {"htn", "bp"} & tags_set and bool(row.get("bp_core", False)):
        score += 8

    if {"diet", "pattern", "sodium"} & tags_set:
        if row.get("ceiling_DietPattern") == "high":
            score += 8
        elif row.get("ceiling_DietPattern") in ["medium", "moderate"]:
            score += 4

    if {"physical_activity", "mvpa"} & tags_set:
        if row.get("ceiling_Movement") in ["high", "medium", "moderate"]:
            score += 6

    if {"weight", "obesity", "bmi"} & tags_set and bool(row.get("bmi_core", False)):
        score += 6

    if {"smoking", "tobacco"} & tags_set and row.get("ceiling_Smoking") == "high":
        score += 5

    if {"monitoring", "followup"} & tags_set and row.get("ceiling_Monitoring") in ["high", "medium", "moderate"]:
        score += 4

    return score


def compute_missing_penalty(row) -> int:
    penalty = 0

    if row.get("feature_tier") == "minimal":
        penalty += MISSING_DATA_PENALTY

    if float(row.get("data_completeness_score", 1.0)) < 0.75:
        penalty += MISSING_DATA_PENALTY

    if row.get("cold_start_flag", False):
        penalty += 5

    return penalty


def score_candidates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["severity_normalized"] = out["severity"].apply(normalize_severity)
    out["base_severity_score"] = out["severity_normalized"].map(SEVERITY_SCORE_MAP).fillna(0).astype(float)

    out["priority_num"] = pd.to_numeric(out["priority"], errors="coerce").fillna(0.0)
    out["base_action_score"] = out["base_severity_score"] + out["priority_num"].clip(lower=0, upper=500) / 10.0

    out["evidence_strength"] = out["evidence"].apply(count_passed_evidence).astype(float) * EVIDENCE_BONUS_PER_PASSED
    out["domain_alignment"] = out.apply(compute_domain_alignment, axis=1).astype(float)
    out["missing_penalty"] = out.apply(compute_missing_penalty, axis=1).astype(float)

    if "contraindications_unknown" in out.columns:
        out["contraindication_penalty"] = (
            out["contraindications_unknown"]
            .fillna(False)
            .astype(bool)
            .map({True: CONTRAINDICATION_UNKNOWN_PENALTY, False: 0})
            .astype(float)
        )
    else:
        out["contraindication_penalty"] = 0.0

    out["total_score"] = (
        out["base_action_score"]
        + out["evidence_strength"]
        + out["domain_alignment"]
        - out["missing_penalty"]
        - out["contraindication_penalty"]
    )

    return out
