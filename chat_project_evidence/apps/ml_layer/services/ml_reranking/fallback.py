import pandas as pd

from .contracts import ensure_ml_output_columns


def rules_only_fallback(
    ranked_df: pd.DataFrame,
    reason: str,
    model_version: str = "unavailable",
    schema_version: str = "unavailable",
    calibration_version: str = "unavailable",
) -> pd.DataFrame:
    """
    Safe fallback:
    - preserve original ranked rows
    - preserve original total_score for downstream compatibility
    - set final_score = total_score
    - add ML metadata columns
    """
    out = ranked_df.copy()

    out["ml_score"] = 0.0
    out["lambda_u"] = 0.0
    out["ml_contrib"] = 0.0
    out["rule_score_norm_participant"] = pd.to_numeric(
        out.get("score_norm", 0.0), errors="coerce"
    ).fillna(0.0)
    out["fusion_rule_component"] = out["rule_score_norm_participant"]
    out["fusion_ml_component"] = 0.0
    out["fusion_strategy"] = "rules_only_fallback"
    out["fusion_version"] = "fallback"
    out["fusion_status"] = "fallback"
    out["fusion_artifact_version"] = None
    if "score_norm" in out.columns:
        out["final_score"] = pd.to_numeric(out["score_norm"], errors="coerce").fillna(0.0)
    else:
        out["final_score"] = 0.0
    out["rerank_status"] = "rules_only_fallback"
    out["fallback_reason"] = reason
    out["ml_model_version"] = model_version
    out["ml_schema_version"] = schema_version
    out["ml_calibration_version"] = calibration_version

    sort_cols = [c for c in ["total_score", "confidence_score", "priority"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(
            by=sort_cols,
            ascending=[False] * len(sort_cols),
            kind="stable",
        ).reset_index(drop=True)

    return ensure_ml_output_columns(out)
