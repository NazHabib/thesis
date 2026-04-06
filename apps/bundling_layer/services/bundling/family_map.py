def infer_bundle_family(action_codes: list[str], tags: list[str]) -> str:
    codes = [str(c or "").upper() for c in action_codes]
    tag_set = {str(t).lower() for t in (tags or [])}

    for code in codes:
        # Internal / system-like messages
        if code.startswith("INFORM_SYSTEM_SCOPE"):
            return "Monitoring"

        # Data quality / missing data
        if code.startswith("INFORM_DATA_MISSING_KEY_METRICS"):
            return "DataQuality"

        # Diabetes self-management
        if code.startswith("RECOMMEND_DIABETES_SELF_MANAGEMENT_CORE"):
            return "DiabetesSelfManagement"

        # Weight / integrated lifestyle-risk style items
        if code.startswith("RECOMMEND_LIFESTYLE_RISK_BUNDLE"):
            return "Weight"
        if code.startswith("INFORM_CENTRAL_OBESITY_RISK"):
            return "Weight"
        if code.startswith("INFORM_NORMAL_GLYCEMIA_PREVENTION"):
            return "Weight"

        # Monitoring
        if code.startswith("ESCALATE_"):
            return "Monitoring"
        if code.startswith("RECOMMEND_BP_"):
            return "Monitoring"
        if code.startswith("RECOMMEND_DIABETES_REVIEW"):
            return "Monitoring"
        if code.startswith("RECOMMEND_HTN_"):
            return "Monitoring"
        if code.startswith("INFORM_BP_"):
            return "Monitoring"
        if code.startswith("INFORM_A1C_"):
            return "Monitoring"
        if code.startswith("RECOMMEND_FOLLOWUP"):
            return "Monitoring"
        if code.startswith("RECOMMEND_RECHECK"):
            return "Monitoring"
        if code.startswith("RECOMMEND_MONITOR"):
            return "Monitoring"

        # Diet
        if code.startswith("INFORM_VERY_HIGH_SODIUM"):
            return "DietPattern"
        if code.startswith("RECOMMEND_SODIUM"):
            return "DietPattern"
        if code.startswith("RECOMMEND_DASH"):
            return "DietPattern"
        if code.startswith("RECOMMEND_MEDITERRANEAN"):
            return "DietPattern"
        if code.startswith("RECOMMEND_DIET"):
            return "DietPattern"
        if code.startswith("INFORM_SODIUM"):
            return "DietPattern"

        # Movement
        if code.startswith("RECOMMEND_MVPA"):
            return "Movement"
        if code.startswith("RECOMMEND_ACTIVITY"):
            return "Movement"
        if code.startswith("RECOMMEND_EXERCISE"):
            return "Movement"

        # Smoking
        if code.startswith("RECOMMEND_SMOKING"):
            return "Smoking"
        if code.startswith("STOP_SMOKING"):
            return "Smoking"

        # Weight
        if code.startswith("RECOMMEND_WEIGHT"):
            return "Weight"
        if code.startswith("RECOMMEND_PREDIABETES"):
            return "Weight"
        if code.startswith("RECOMMEND_BMI"):
            return "Weight"
        if code.startswith("RECOMMEND_OBESITY"):
            return "Weight"
        if code.startswith("RECOMMEND_CALORIE"):
            return "Weight"

        # Lipids
        if code.startswith("RECOMMEND_LIPID"):
            return "Lipids"
        if code.startswith("RECOMMEND_CHOLESTEROL"):
            return "Lipids"
        if code.startswith("RECOMMEND_HDL"):
            return "Lipids"
        if code.startswith("RECOMMEND_TC_"):
            return "Lipids"
        if code.startswith("INFORM_LIPID"):
            return "Lipids"
        if code.startswith("INFORM_CHOLESTEROL"):
            return "Lipids"

        # Alcohol
        if code.startswith("RECOMMEND_ALCOHOL"):
            return "Alcohol"
        if code.startswith("INFORM_ALCOHOL"):
            return "Alcohol"

    # tag-based fallback
    if {"data_quality", "missingness"} & tag_set:
        return "DataQuality"

    if {"diabetes", "self_management"} & tag_set:
        return "DiabetesSelfManagement"

    if {"acute", "safety", "monitoring", "followup", "a1c", "bp", "htn"} & tag_set:
        return "Monitoring"

    if {"diet", "pattern", "sodium", "fiber", "satfat", "nutrition"} & tag_set:
        return "DietPattern"

    if {"physical_activity", "mvpa", "sedentary", "exercise"} & tag_set:
        return "Movement"

    if {"smoking", "tobacco", "cessation"} & tag_set:
        return "Smoking"

    if {
        "weight",
        "obesity",
        "bmi",
        "prediabetes",
        "calorie",
        "central_obesity",
        "risk_marker",
        "glycemia",
        "prevention",
        "lifestyle",
        "risk",
    } & tag_set:
        return "Weight"

    if {"lipid", "cholesterol", "hdl", "tc"} & tag_set:
        return "Lipids"

    if {"alcohol"} & tag_set:
        return "Alcohol"

    return "IntegratedCardiometabolic"
