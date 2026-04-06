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

    if "total_score" in df.columns:
        try:
            pd.to_numeric(df["total_score"])
        except Exception:
            issues.append("total_score is not numeric")

    if "confidence_score" in df.columns:
        try:
            pd.to_numeric(df["confidence_score"])
        except Exception:
            issues.append("confidence_score is not numeric")

    return len(issues) == 0, issues
