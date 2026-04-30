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

    # ---- Null checks ----
    if "ParticipantKey" in df.columns and df["ParticipantKey"].isna().any():
        issues.append("ParticipantKey contains null values")

    if "action_code" in df.columns and df["action_code"].isna().any():
        issues.append("action_code contains null values")

    if "bundle_family" in df.columns and df["bundle_family"].isna().any():
        issues.append("bundle_family contains null values")

    # ---- Feature tier ----
    allowed_feature_tiers = {"minimal", "core", "full"}
    if "feature_tier" in df.columns:
        invalid = sorted(set(df["feature_tier"].dropna().astype(str)) - allowed_feature_tiers)
        if invalid:
            issues.append(f"feature_tier contains invalid values: {invalid}")

    # ---- Numeric checks ----
    numeric_cols = [
        "total_score",
        "score_norm",
        "confidence_score",
        "data_completeness_score",
        "collab_weighted_support_rate",
        "collab_weighted_mean_rule_score_norm",
        "collab_weighted_mean_rank_pre_ml",
        "collab_mean_supporting_similarity",

        # collaborative
        "collab_available_flag",
        "collab_neighbor_count",
        "collab_support_count",
        "collab_support_rate",
        "collab_mean_rule_score_norm",
        "collab_mean_rank_pre_ml",
        "collab_family_support_rate",
    ]

    for col in numeric_cols:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            if s.isna().any():
                issues.append(f"{col} contains non-numeric or missing values")

    # ---- Range checks ----
    bounded_cols = [
        "score_norm",
        "confidence_score",
        "data_completeness_score",
        "collab_available_flag",
        "collab_support_rate",
        "collab_mean_rule_score_norm",
        "collab_family_support_rate",
        "collab_weighted_support_rate",
        "collab_weighted_mean_rule_score_norm",
        "collab_mean_supporting_similarity",
    ]

    for col in bounded_cols:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            if s.notna().any() and ((s < 0) | (s > 1)).any():
                issues.append(f"{col} contains values outside expected range [0, 1]")

    # ---- Duplicate check ----
    if {"ParticipantKey", "action_code"}.issubset(df.columns):
        dupes = int(df.duplicated(subset=["ParticipantKey", "action_code"]).sum())
        if dupes > 0:
            issues.append(f"Duplicate ParticipantKey/action_code rows detected: {dupes}")

    return len(issues) == 0, issues
