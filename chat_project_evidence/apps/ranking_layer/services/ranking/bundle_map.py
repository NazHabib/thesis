import pandas as pd


ACTION_CODE_PREFIX_MAP = {
    "RECOMMEND_DASH": "DietPattern",
    "RECOMMEND_MEDITERRANEAN": "DietPattern",
    "RECOMMEND_DIABETES_REVIEW": "Monitoring",
    "INFORM_VERY_HIGH_SODIUM": "DietPattern",
    "RECOMMEND_MVPA": "Movement",
    "RECOMMEND_BP_": "Monitoring",
    "ESCALATE_": "Monitoring",
    "STOP_SMOKING": "Smoking",
    "RECOMMEND_SMOKING": "Smoking",
    "RECOMMEND_ALCOHOL": "Alcohol",
    "RECOMMEND_WEIGHT": "Weight",
    "RECOMMEND_LIPIDS": "Lipids",
}


def infer_bundle_family(action_code: str, tags) -> str:
    action_code = str(action_code or "").upper()
    tags = tags if isinstance(tags, list) else []
    tag_set = {str(t).lower() for t in tags}

    for prefix, family in ACTION_CODE_PREFIX_MAP.items():
        if action_code.startswith(prefix):
            return family

    if {"diet", "pattern", "sodium"} & tag_set:
        return "DietPattern"
    if {"physical_activity", "mvpa", "sedentary"} & tag_set:
        return "Movement"
    if {"weight", "obesity", "bmi"} & tag_set:
        return "Weight"
    if {"smoking", "tobacco"} & tag_set:
        return "Smoking"
    if {"alcohol"} & tag_set:
        return "Alcohol"
    if {"lipid", "cholesterol", "hdl"} & tag_set:
        return "Lipids"
    if {"monitoring", "followup", "a1c", "bp"} & tag_set:
        return "Monitoring"
    if {"acute", "safety"} & tag_set:
        return "Monitoring"

    return "IntegratedCardiometabolic"


def apply_bundle_family(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Force object/string-compatible dtype first
    if "bundle_family" not in out.columns:
        out["bundle_family"] = pd.Series([None] * len(out), index=out.index, dtype="object")
    else:
        out["bundle_family"] = out["bundle_family"].astype("object")

    missing_mask = out["bundle_family"].isna() | (out["bundle_family"].astype("string").str.strip() == "")

    inferred = out.loc[missing_mask].apply(
        lambda row: infer_bundle_family(row.get("action_code"), row.get("tags")),
        axis=1,
    )

    out.loc[missing_mask, "bundle_family"] = inferred.astype("object")

    return out
