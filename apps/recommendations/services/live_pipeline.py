from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from django.conf import settings

from apps.accounts.models import User
from apps.profiles.services import build_current_health_feature_payload
from apps.recs_csl.services.cold_start.service import run_cold_start_layer
from apps.ranking_layer.services.ranking.service import run_ranking_layer
from apps.ml_layer.services.ml_reranking.service import run_ml_reranking_layer
from apps.ml_layer.services.ml_reranking.adapter import prepare_reranked_for_bundling
from apps.bundling_layer.services.bundling.service import run_bundling_layer
from apps.explainability_layer.services.explainability.service import run_explainability_layer

from rules_safety_engine.rules_engine.engine import evaluate_rules
from rules_safety_engine.rules_engine.rule_loader import load_rules_from_yaml
from rules_safety_engine.rules_engine.utils import to_plain_dict


@dataclass
class PatientSafeRecommendation:
    title: str
    category: str
    priority: int
    rationale: str
    explanation: str
    source: str
    confidence: float


@dataclass
class LivePipelineResult:
    recommendations: list[PatientSafeRecommendation]
    manifest: dict[str, Any]
    explained_df: pd.DataFrame


def run_live_patient_pipeline(
    user: User,
    fusion_strategy: str | None = None,
) -> LivePipelineResult:
    """
    Run the full thesis hybrid pipeline for one live patient/user.

    Input:
    - PatientProfile
    - PatientHealthRecord timeline

    Pipeline:
    - rules engine
    - cold-start safety/context
    - ranking
    - ML reranking + bounded fusion
    - bundling
    - explainability

    Output:
    - patient-safe recommendation objects for the Django patient portal
    """
    base_dir = Path(settings.BASE_DIR)

    participant_key = f"user_{user.id}_live"

    cfm_df = _build_live_cfm_df(user=user, participant_key=participant_key)

    rules = _load_rules(base_dir)
    rules_result = evaluate_rules(
        df=cfm_df,
        rules=rules,
        input_path="patient_entered_live",
        engine_version="live_single_patient",
    )

    participants_df = _build_participants_candidates_df(
        participant_key=participant_key,
        recommendations=rules_result.recommendations,
    )

    if participants_df.empty:
        raise ValueError("No rule-based candidates were produced for this live patient input.")

    cold_start_out = run_cold_start_layer(
        cfm_df=cfm_df,
        participants_df=participants_df,
    )
    safety_context_df = cold_start_out["safety_context"]
    gated_candidates_df = cold_start_out["gated_candidates"]

    if gated_candidates_df.empty:
        raise ValueError("Cold-start layer removed all candidates for this live patient input.")

    ranking_out = run_ranking_layer(
        safety_context_df=safety_context_df,
        gated_candidates_df=gated_candidates_df,
    )
    ranked_df = ranking_out["ranked_recommendations"]

    ml_out = run_ml_reranking_layer(
        safety_context_df=safety_context_df,
        ranked_df=ranked_df,
        participant_features_df=cfm_df,
        fusion_strategy=fusion_strategy,
    )
    reranked_df = ml_out["reranked_recommendations"]

    bundling_input_df = prepare_reranked_for_bundling(reranked_df)

    bundling_out = run_bundling_layer(
        safety_context_df=safety_context_df,
        ranked_df=bundling_input_df,
    )
    bundles_df = bundling_out["final_recommendations"]

    explainability_out = run_explainability_layer(
        safety_context_df=safety_context_df,
        final_recommendations_df=bundles_df,
        reranked_df=reranked_df,
        participant_features_df=cfm_df,
    )
    explained_df = explainability_out["explained_recommendations"]

    patient_safe_recs = _patient_safe_recommendations_from_explained(explained_df)

    if not patient_safe_recs:
        raise ValueError("Explainability layer produced no patient-facing recommendations.")

    manifest = {
        "pipeline_mode": "live_single_patient_pipeline",
        "participant_key": participant_key,
        "rules": {
            "rules_loaded": len(rules),
            "recommendations": len(rules_result.recommendations),
            "fired_rules": len(rules_result.fired_rules),
            "conflicts": len(rules_result.conflicts),
            "warnings": list(rules_result.warnings),
        },
        "cold_start": cold_start_out.get("manifest", {}),
        "ranking": ranking_out.get("manifest", {}),
        "ml_reranking": ml_out.get("manifest", {}),
        "bundling": bundling_out.get("manifest", {}),
        "explainability": explainability_out.get("manifest", {}),
        "rows": {
            "cfm_rows": int(len(cfm_df)),
            "participants_rows": int(len(participants_df)),
            "gated_candidates": int(len(gated_candidates_df)),
            "ranked": int(len(ranked_df)),
            "reranked": int(len(reranked_df)),
            "bundles": int(len(bundles_df)),
            "explained": int(len(explained_df)),
        },
    }

    return LivePipelineResult(
        recommendations=patient_safe_recs,
        manifest=manifest,
        explained_df=explained_df,
    )


def _load_rules(base_dir: Path):
    rules_path = base_dir / "rules_safety_engine" / "rules" / "example_rules.yaml"
    schema_path = base_dir / "rules_safety_engine" / "rules" / "rule_schema.json"

    return load_rules_from_yaml(
        rule_path=rules_path,
        schema_path=schema_path,
    )


def _build_live_cfm_df(user: User, participant_key: str) -> pd.DataFrame:
    payload = build_current_health_feature_payload(user)

    row = {
        key: value
        for key, value in payload.items()
        if not str(key).startswith("_")
    }

    row["ParticipantKey"] = participant_key
    row["Cycle"] = "patient_entered_live"

    _add_required_defaults(row)
    _add_rule_engine_derived_fields(row)

    return pd.DataFrame([row])


def _add_required_defaults(row: dict[str, Any]) -> None:
    """
    Ensure the live row satisfies the required cold-start columns and common
    rule-engine condition fields.
    """
    defaults = {
        "SBP_mean": None,
        "DBP_mean": None,
        "BP_n_used": None,
        "LBXGH": None,
        "BMXBMI": None,
        "DietDay1_reliable_flag": 0,
        "DR1TSODI": None,
        "Current_smoker_flag": 0,
        "Any_MVPA_flag": 0,
        "Hypertensive_crisis_flag": 0,
        "Severe_hyperglycemia_flag": 0,
        "Hypertension": 0,
        "Diabetes": 0,
        "Prediabetes": 0,
        "Severe_obesity_flag": 0,
        "Sedentary_high_flag": 0,
        "Sodium_over_2300_flag": 0,
        "Sodium_over_1500_flag": 0,
        "SatFat_over_10pct_flag": 0,
        "HDL_low_flag": 0,
        "TC_high_flag": 0,
        "Comorbidity": 0,
        "HTN_severity_score": 0,
        "DM_severity_score": 0,
        "Obesity_severity_score": 0,
        "Lifestyle_risk_score": 0,
        "Cardiometabolic_severity_score": 0,
        "Severity_band": "minimal",
        "data_completeness_score": 0.0,
        "SEQN": None,
        "Smoking_status": "unknown",
        "Potassium_to_Sodium_ratio": None,
        "PAD680": None,
        "Central_obesity_flag_atp3": 0,
        "BP_method_inferred": "patient_entered",
        "BP_method": "patient_entered",
        "Diet_day1_mode": "patient_entered",
        "SatFat_energy_pct": None,
    }

    for key, value in defaults.items():
        row.setdefault(key, value)


def _add_rule_engine_derived_fields(row: dict[str, Any]) -> None:
    age = _safe_float(row.get("RIDAGEYR"))
    bmi = _safe_float(row.get("BMXBMI"))
    hba1c = _safe_float(row.get("LBXGH"))
    sbp = _safe_float(row.get("SBP_mean"))
    dbp = _safe_float(row.get("DBP_mean"))
    sodium = _safe_float(row.get("DR1TSODI"))
    fibre = _safe_float(row.get("DR1TFIBE"))
    sugar = _safe_float(row.get("DR1TSUGR"))
    calories = _safe_float(row.get("DR1TKCAL"))
    satfat_g = _safe_float(row.get("DR1TSFAT"))
    hdl = _safe_float(row.get("HDL_mg_dL"))
    total_chol = _safe_float(row.get("TotalChol_mg_dL"))

    row["Adult18plus_flag"] = int(age is not None and age >= 18)
    row["Adult20plus_flag"] = int(age is not None and age >= 20)

    row["DR1DRSTZ"] = 1 if int(row.get("DietDay1_reliable_flag") or 0) == 1 else 0

    if bmi is not None:
        if bmi >= 30:
            row["BMI_category"] = "obesity"
        elif bmi >= 25:
            row["BMI_category"] = "overweight"
        elif bmi >= 18.5:
            row["BMI_category"] = "normal"
        else:
            row["BMI_category"] = "underweight"

    if hba1c is not None:
        row["Prediabetes"] = int(5.7 <= hba1c < 6.5 and not bool(row.get("Diabetes")))
        row["Diabetes"] = max(int(row.get("Diabetes") or 0), int(hba1c >= 6.5))
        row["Severe_hyperglycemia_flag"] = max(
            int(row.get("Severe_hyperglycemia_flag") or 0),
            int(hba1c >= 10.0),
        )

    if sbp is not None or dbp is not None:
        sbp_value = sbp or 0
        dbp_value = dbp or 0

        row["Hypertension"] = max(
            int(row.get("Hypertension") or 0),
            int(sbp_value >= 130 or dbp_value >= 80),
        )
        row["Hypertensive_crisis_flag"] = max(
            int(row.get("Hypertensive_crisis_flag") or 0),
            int(sbp_value >= 180 or dbp_value >= 120),
        )

    if bmi is not None:
        row["Severe_obesity_flag"] = int(bmi >= 40)
        row["obesity_flag_proxy"] = int(bmi >= 30)
    
    waist = _safe_float(row.get("BMXWAIST"))
    sex_code = _safe_float(row.get("RIAGENDR"))

    if waist is not None and sex_code is not None:
        if int(sex_code) == 1:
            row["Central_obesity_flag_atp3"] = int(waist > 102)
        elif int(sex_code) == 2:
            row["Central_obesity_flag_atp3"] = int(waist > 88)

    if sodium is not None:
        row["Sodium_over_2300_flag"] = int(sodium > 2300)
        row["Sodium_over_1500_flag"] = int(sodium > 1500)

    if calories is not None and calories > 0 and sodium is not None:
        row["Sodium_mg_per_1000kcal"] = round(sodium / calories * 1000, 3)

    if calories is not None and calories > 0 and fibre is not None:
        row["Fiber_g_per_1000kcal"] = round(fibre / calories * 1000, 3)
        
    potassium = _safe_float(row.get("DR1TPOTA"))

    if potassium is not None and sodium is not None and sodium > 0:
        row["Potassium_to_Sodium_ratio"] = round(potassium / sodium, 3)

    if calories is not None and calories > 0 and satfat_g is not None:
        satfat_energy_pct = satfat_g * 9 / calories * 100
        row["SatFat_energy_pct"] = round(satfat_energy_pct, 3)
        row["SatFat_over_10pct_flag"] = int(satfat_energy_pct > 10)
    else:
        row.setdefault("SatFat_energy_pct", None)
        row.setdefault("SatFat_over_10pct_flag", 0)

    if total_chol is not None:
        row["TC_high_flag"] = int(total_chol >= 240)
        row["TC_borderline_flag"] = int(200 <= total_chol < 240)

    if hdl is not None:
        row["HDL_low_flag"] = int(hdl < 40)

    if total_chol is not None and hdl is not None and hdl > 0:
        row["TC_to_HDL_ratio"] = round(total_chol / hdl, 3)

    if sugar is not None:
        row["Sugar_g_day"] = sugar
        
    if int(row.get("Current_smoker_flag") or 0) == 1:
        row["Smoking_status"] = "current"
    else:
        row["Smoking_status"] = "not_current_or_unknown"

    row["Comorbidity"] = int(
        sum(
            [
                int(row.get("Hypertension") or 0),
                int(row.get("Diabetes") or 0),
                int(row.get("Severe_obesity_flag") or 0),
                int(row.get("TC_high_flag") or 0),
            ]
        )
        >= 2
    )


def _safe_float(value: Any) -> float | None:
    if value in [None, ""]:
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if pd.isna(number):
        return None

    return number


def _build_participants_candidates_df(
    participant_key: str,
    recommendations: list[Any],
) -> pd.DataFrame:
    candidates = []

    for rec in recommendations:
        rec_dict = to_plain_dict(rec)

        candidates.append(
            {
                "action_code": rec_dict.get("action_code"),
                "action_type": rec_dict.get("action_type"),
                "severity": rec_dict.get("severity"),
                "priority": rec_dict.get("priority"),
                "tags": rec_dict.get("tags", []),
                "evidence": rec_dict.get("evidence", []),
                "source_rule_ids": rec_dict.get("source_rule_ids", []),
                "bundle_family": _infer_bundle_family(
                    action_code=rec_dict.get("action_code"),
                    action_type=rec_dict.get("action_type"),
                    tags=rec_dict.get("tags", []),
                ),
            }
        )

    if not candidates:
        return pd.DataFrame(columns=["ParticipantKey", "Candidates"])

    return pd.DataFrame(
        [
            {
                "ParticipantKey": participant_key,
                "Candidates": candidates,
            }
        ]
    )


def _infer_bundle_family(action_code, action_type, tags) -> str:
    action_code = str(action_code or "").upper()
    action_type = str(action_type or "").lower()
    tags = tags if isinstance(tags, list) else []
    tags_upper = {str(t).upper() for t in tags}

    if (
        "ESCALATE" in action_code
        or action_type == "escalate"
        or "ACUTE" in tags_upper
        or "SAFETY" in tags_upper
    ):
        return "Monitoring"

    if (
        "DIET" in action_code
        or "DASH" in action_code
        or "MEDITERRANEAN" in action_code
        or "SODIUM" in action_code
        or "FIBER" in action_code
        or "FIBRE" in action_code
        or "DIET" in tags_upper
        or "SODIUM" in tags_upper
        or "PATTERN" in tags_upper
        or "NHANES_DAY1" in tags_upper
    ):
        return "DietPattern"

    if (
        "MVPA" in action_code
        or "ACTIVITY" in action_code
        or "WALK" in action_code
        or "EXERCISE" in action_code
        or "PA-" in action_code
        or "PHYSICAL_ACTIVITY" in tags_upper
        or "MVPA" in tags_upper
    ):
        return "Movement"

    if (
        "WEIGHT" in action_code
        or "OBESITY" in action_code
        or "BMI" in action_code
        or "WAIST" in action_code
    ):
        return "Weight"

    if (
        "SMOK" in action_code
        or "TOBACCO" in action_code
        or "SMOKING" in tags_upper
    ):
        return "Smoking"

    if (
        "ALCOHOL" in action_code
        or "DRINK" in action_code
        or "ALCOHOL" in tags_upper
    ):
        return "Alcohol"

    if (
        "LIPID" in action_code
        or "CHOLESTEROL" in action_code
        or "HDL" in action_code
        or "TC_" in action_code
        or "LIPIDS" in tags_upper
    ):
        return "Lipids"

    if (
        "MONITOR" in action_code
        or "REVIEW" in action_code
        or "FOLLOWUP" in action_code
        or "FOLLOW_UP" in action_code
        or "BP_" in action_code
        or "A1C" in action_code
        or "FOLLOWUP" in tags_upper
        or "A1C" in tags_upper
        or "DIABETES" in tags_upper
        or "HTN" in tags_upper
    ):
        return "Monitoring"

    if "CARDIOMETABOLIC" in action_code or "INTEGRATED" in action_code:
        return "IntegratedCardiometabolic"

    return "Monitoring"


def _patient_safe_recommendations_from_explained(
    explained_df: pd.DataFrame,
) -> list[PatientSafeRecommendation]:
    if explained_df.empty:
        return []

    sort_cols = [
        col
        for col in [
            "followup_window_order",
            "bundle_rank",
            "bundle_score",
            "confidence_score",
            "priority",
        ]
        if col in explained_df.columns
    ]

    if sort_cols:
        ascending = []
        for col in sort_cols:
            ascending.append(col in {"followup_window_order", "bundle_rank"})

        explained_df = explained_df.sort_values(
            by=sort_cols,
            ascending=ascending,
            kind="stable",
        )

    recs: list[PatientSafeRecommendation] = []

    for rank, row in enumerate(explained_df.head(5).to_dict(orient="records"), start=1):
        title = _safe_text(row.get("bundle_title"), "Health recommendation")
        category = _safe_text(row.get("category"), "guidance")
        rationale = _safe_text(
            row.get("rationale"),
            "This recommendation was selected from the available health information.",
        )

        patient_explanation = _safe_text(row.get("patient_facing_explanation"))
        confidence_explanation = _safe_text(row.get("patient_confidence_explanation"))
        followup_text = _safe_text(row.get("followup_timing_text"))

        explanation_parts = []

        if patient_explanation:
            explanation_parts.append(patient_explanation)

        if confidence_explanation and confidence_explanation not in patient_explanation:
            explanation_parts.append(confidence_explanation)

        if followup_text and followup_text not in patient_explanation:
            explanation_parts.append(followup_text)

        explanation = " ".join(explanation_parts).strip()

        if not explanation:
            explanation = (
                "This recommendation was generated from your saved health data. "
                "Please review it with a healthcare professional if you are unsure."
            )

        recs.append(
            PatientSafeRecommendation(
                title=title,
                category=category,
                priority=_patient_priority_from_rank(rank),
                rationale=rationale,
                explanation=explanation,
                source="live_pipeline",
                confidence=float(row.get("confidence_score") or 0.0),
            )
        )

    return recs


def _patient_priority_from_rank(rank: int) -> int:
    if rank <= 1:
        return 1
    if rank <= 3:
        return 2
    return 3


def _safe_text(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback

    text = str(value).strip()
    return text if text else fallback
