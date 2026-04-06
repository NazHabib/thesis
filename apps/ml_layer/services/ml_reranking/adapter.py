import pandas as pd


def prepare_reranked_for_bundling(reranked_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare ML reranked output for bundling without losing traceability.

    Policy:
    - preserve original total_score
    - expose effective_score for downstream use
    - overwrite total_score with final_score only in the bundling input copy
      so existing bundling logic can work with minimal changes
    """
    out = reranked_df.copy()

    if "final_score" in out.columns:
        out["rule_total_score"] = out["total_score"]
        out["effective_score"] = out["final_score"]
        out["total_score"] = out["final_score"]

    return out
