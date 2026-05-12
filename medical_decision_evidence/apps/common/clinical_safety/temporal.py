from __future__ import annotations

import pandas as pd


TEMPORAL_FEATURE_COLUMNS = [
    "followup_urgency",
    "followup_horizon_days",
    "monitoring_need_level",
    "temporal_risk_tier",
    "persistence_risk_proxy",
    "repeat_measurement_available",
    "temporal_data_quality",
    "longitudinal_ready_flag",
]


def build_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    sbp = pd.to_numeric(out["SBP_mean"], errors="coerce") if "SBP_mean" in out.columns else pd.Series(index=out.index, dtype="float64")
    dbp = pd.to_numeric(out["DBP_mean"], errors="coerce") if "DBP_mean" in out.columns else pd.Series(index=out.index, dtype="float64")
    hba1c = pd.to_numeric(out["LBXGH"], errors="coerce") if "LBXGH" in out.columns else pd.Series(index=out.index, dtype="float64")
    bmi = pd.to_numeric(out["BMXBMI"], errors="coerce") if "BMXBMI" in out.columns else pd.Series(index=out.index, dtype="float64")
    bp_n = pd.to_numeric(out["BP_n_used"], errors="coerce") if "BP_n_used" in out.columns else pd.Series(index=out.index, dtype="float64")

    crisis_flag = out["Hypertensive_crisis_flag"].fillna(0).astype(float) if "Hypertensive_crisis_flag" in out.columns else pd.Series(0.0, index=out.index)
    severe_hyper_flag = out["Severe_hyperglycemia_flag"].fillna(0).astype(float) if "Severe_hyperglycemia_flag" in out.columns else pd.Series(0.0, index=out.index)
    diet_reliable = out["DietDay1_reliable_flag"].fillna(0).astype(float) if "DietDay1_reliable_flag" in out.columns else pd.Series(0.0, index=out.index)

    out["repeat_measurement_available"] = (bp_n >= 2).fillna(False).astype(int)

    out["persistence_risk_proxy"] = (
        (sbp >= 140).fillna(False).astype(int)
        + (dbp >= 90).fillna(False).astype(int)
        + (hba1c >= 7.0).fillna(False).astype(int)
        + (bmi >= 30.0).fillna(False).astype(int)
    )

    followup_urgency = pd.Series("routine", index=out.index, dtype="object")

    urgent_mask = (
        (crisis_flag == 1.0)
        | (severe_hyper_flag == 1.0)
        | (sbp >= 160).fillna(False)
        | (dbp >= 100).fillna(False)
        | (hba1c >= 9.0).fillna(False)
    )

    soon_mask = (
        (sbp >= 140).fillna(False)
        | (dbp >= 90).fillna(False)
        | (hba1c >= 7.0).fillna(False)
        | (bmi >= 35.0).fillna(False)
    ) & ~urgent_mask

    monitor_mask = (
        (sbp >= 130).fillna(False)
        | (dbp >= 80).fillna(False)
        | (hba1c >= 6.5).fillna(False)
        | (bmi >= 30.0).fillna(False)
    ) & ~urgent_mask & ~soon_mask

    followup_urgency.loc[monitor_mask] = "monitor"
    followup_urgency.loc[soon_mask] = "soon"
    followup_urgency.loc[urgent_mask] = "urgent"

    out["followup_urgency"] = followup_urgency

    horizon_map = {
        "urgent": 30,
        "soon": 90,
        "monitor": 180,
        "routine": 365,
    }
    out["followup_horizon_days"] = out["followup_urgency"].map(horizon_map).astype("Int64")

    monitoring_need = pd.Series("low", index=out.index, dtype="object")
    monitoring_need.loc[monitor_mask] = "moderate"
    monitoring_need.loc[soon_mask] = "high"
    monitoring_need.loc[urgent_mask] = "very_high"
    out["monitoring_need_level"] = monitoring_need

    temporal_risk_tier = pd.Series("stable", index=out.index, dtype="object")
    temporal_risk_tier.loc[out["persistence_risk_proxy"] >= 1] = "elevated"
    temporal_risk_tier.loc[out["persistence_risk_proxy"] >= 2] = "high"
    temporal_risk_tier.loc[urgent_mask] = "acute"
    out["temporal_risk_tier"] = temporal_risk_tier

    temporal_quality = pd.Series("limited", index=out.index, dtype="object")
    temporal_quality.loc[(out["repeat_measurement_available"] == 1) | (diet_reliable == 1.0)] = "usable"
    temporal_quality.loc[(out["repeat_measurement_available"] == 1) & (diet_reliable == 1.0)] = "good"
    out["temporal_data_quality"] = temporal_quality

    out["longitudinal_ready_flag"] = (
        (out["repeat_measurement_available"] == 1) | (diet_reliable == 1.0)
    ).astype(int)

    return out
