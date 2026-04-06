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
    out["final_score"] = out["total_score"]
    out["rerank_status"] = "rules_only_fallback"
    out["fallback_reason"] = reason
    out["ml_model_version"] = model_version
    out["ml_schema_version"] = schema_version
    out["ml_calibration_version"] = calibration_version

    # keep row order stable but ensure score ordering if needed
    out = out.sort_values(
        by=["total_score", "confidence_score", "priority"],
        ascending=[False, False, False],
        kind="stable",
    ).reset_index(drop=True)

    return ensure_ml_output_columns(out)
