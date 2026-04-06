import numpy as np
import pandas as pd

from .config import CONFIDENCE_TO_NUM, NUM_TO_CONFIDENCE


def normalize_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    group_min = out.groupby("ParticipantKey", sort=False)["total_score"].transform("min")
    group_max = out.groupby("ParticipantKey", sort=False)["total_score"].transform("max")

    denom = (group_max - group_min).replace(0, np.nan)
    out["score_norm"] = ((out["total_score"] - group_min) / denom).fillna(0.5)

    return out


def score_to_confidence_label(score_norm: float) -> str:
    if score_norm >= 0.80:
        return "high"
    if score_norm >= 0.50:
        return "medium"
    return "low"


def apply_confidence(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_scores(df)

    out["confidence_label_raw"] = out["score_norm"].apply(score_to_confidence_label)

    raw_num = out["confidence_label_raw"].map(CONFIDENCE_TO_NUM).fillna(1).astype(int)
    ceiling_num = out["coldstart_confidence_ceiling"].map(CONFIDENCE_TO_NUM).fillna(2).astype(int)

    out["confidence_num_final"] = np.minimum(raw_num.to_numpy(), ceiling_num.to_numpy())
    out["confidence_label"] = out["confidence_num_final"].map(NUM_TO_CONFIDENCE)
    out["confidence_score"] = out["score_norm"].round(4)

    return out
