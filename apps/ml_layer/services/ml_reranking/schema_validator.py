import pandas as pd


def validate_ml_feature_table(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """
    Lightweight runtime validator for inference-time feature table.
    """
    issues: list[str] = []

    required = [
        "ParticipantKey",
        "action_code",
        "bundle_family",
        "total_score",
        "score_norm",
        "confidence_score",
        "feature_tier",
        "data_completeness_score",
        "triage_mode",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        issues.append(f"Missing required ML feature columns: {missing}")

    if "ParticipantKey" in df.columns and df["ParticipantKey"].isna().any():
        issues.append("ParticipantKey contains null values")

    if "action_code" in df.columns and df["action_code"].isna().any():
        issues.append("action_code contains null values")

    if "bundle_family" in df.columns and df["bundle_family"].isna().any():
        issues.append("bundle_family contains null values")

    allowed_feature_tiers = {"minimal", "core", "full"}
    if "feature_tier" in df.columns:
        invalid = sorted(set(df["feature_tier"].dropna().astype(str)) - allowed_feature_tiers)
        if invalid:
            issues.append(f"feature_tier contains invalid values: {invalid}")

    for col in ["total_score", "score_norm", "confidence_score", "data_completeness_score"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            if s.isna().any():
                issues.append(f"{col} contains non-numeric or missing values")

    for col in ["score_norm", "confidence_score", "data_completeness_score"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            if s.notna().any() and ((s < 0) | (s > 1)).any():
                issues.append(f"{col} contains values outside expected range [0, 1]")

    if {"ParticipantKey", "action_code"}.issubset(df.columns):
        dupes = int(df.duplicated(subset=["ParticipantKey", "action_code"]).sum())
        if dupes > 0:
            issues.append(f"Duplicate ParticipantKey/action_code rows detected: {dupes}")

    return len(issues) == 0, issues
