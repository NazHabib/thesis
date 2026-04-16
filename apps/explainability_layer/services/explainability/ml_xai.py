from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from django.conf import settings

try:
    import shap
except Exception:  # pragma: no cover
    shap = None

from apps.ml_layer.services.ml_reranking.feature_builder import build_ml_feature_table


ACTIVE_MANIFEST_PATH = (
    Path(settings.BASE_DIR)
    / "apps"
    / "ml_layer"
    / "artifacts"
    / "manifests"
    / "active_manifest.json"
)


FEATURE_LABEL_MAP = {
    "num__score_norm": "Normalized rule score",
    "num__confidence_score": "Upstream confidence score",
    "num__total_score": "Rule-based total score",
    "num__priority": "Recommendation priority",
    "num__domain_alignment": "Domain alignment score",
    "num__base_action_score": "Base action score",
    "num__base_severity_score": "Base severity score",
    "num__evidence_strength": "Evidence strength score",
    "num__missing_penalty": "Missing-data penalty",
    "num__contraindication_penalty": "Contraindication penalty",
    "num__data_completeness_score": "Data completeness score",
    "num__LBXGH": "HbA1c",
    "num__DR1TSODI": "Day-1 sodium intake",
    "num__DR1TFIBE": "Day-1 fibre intake",
    "num__DR1TSUGR": "Day-1 total sugar intake",
    "num__SBP_mean": "Mean systolic blood pressure",
    "num__DBP_mean": "Mean diastolic blood pressure",
    "num__BMXBMI": "Body mass index",
    "num__BMXWAIST": "Waist circumference",
    "num__Current_smoker_flag": "Current smoker flag",
    "num__Any_MVPA_flag": "Any moderate/vigorous activity flag",
    "num__HDL_low_flag": "Low HDL flag",
    "num__TC_high_flag": "High total cholesterol flag",
    "num__DM_severity_score": "Diabetes severity score",
    "num__HTN_severity_score": "Hypertension severity score",
    "num__Obesity_severity_score": "Obesity severity score",
    "num__Lifestyle_risk_score": "Lifestyle risk score",
    "num__Cardiometabolic_severity_score": "Cardiometabolic severity score",
    "num__Alcohol_over_guideline_flag": "Alcohol above guideline flag",
    "num__SatFat_over_10pct_flag": "Saturated fat above 10% energy flag",
    "num__Sodium_over_2300_flag": "Sodium above 2300 mg flag",
    "num__Sodium_over_1500_flag": "Sodium above 1500 mg flag",
    "num__Hypertensive_crisis_flag": "Hypertensive crisis flag",
    "num__Severe_hyperglycemia_flag": "Severe hyperglycaemia flag",
    "cat__confidence_label_high": "High upstream confidence",
    "cat__confidence_label_medium": "Medium upstream confidence",
    "cat__confidence_label_low": "Low upstream confidence",
    "cat__Severity_band_Low": "Low severity band",
    "cat__Severity_band_Moderate": "Moderate severity band",
    "cat__Severity_band_High": "High severity band",
    "cat__bundle_family_DietPattern": "Diet pattern bundle family",
    "cat__bundle_family_Movement": "Movement bundle family",
    "cat__bundle_family_Monitoring": "Monitoring bundle family",
    "cat__bundle_family_Weight": "Weight bundle family",
    "cat__bundle_family_Smoking": "Smoking bundle family",
    "cat__bundle_family_Lipids": "Lipids bundle family",
    "cat__severity_high": "High candidate severity",
    "cat__severity_low": "Low candidate severity",
    "cat__severity_moderate": "Moderate candidate severity",
    "num__is_monitoring_bundle": "Monitoring bundle indicator",
    "num__is_diet_bundle": "Diet bundle indicator",
    "num__is_movement_bundle": "Movement bundle indicator",
    "num__is_weight_bundle": "Weight bundle indicator",
    "num__is_smoking_bundle": "Smoking bundle indicator",
    "cat__action_type_inform": "Informational action type",
    "cat__action_type_recommend": "Recommendation action type",
    "num__Hypertension": "Hypertension flag",
    "num__Diabetes": "Diabetes flag",
    "num__Prediabetes": "Prediabetes flag",
    "cat__action_type_escalate": "Escalation action type",
    "cat__bundle_family_DietPattern": "Diet pattern bundle family",
    "cat__bundle_family_Movement": "Movement bundle family",
    "cat__bundle_family_Monitoring": "Monitoring bundle family",
    "cat__bundle_family_Weight": "Weight bundle family",
    "cat__bundle_family_Smoking": "Smoking bundle family",
    "cat__bundle_family_Lipids": "Lipids bundle family",
}

FEATURE_GROUP_MAP = {
    "score_norm": "meta_ranking",
    "confidence_score": "meta_ranking",
    "total_score": "meta_ranking",
    "priority": "meta_ranking",
    "domain_alignment": "meta_ranking",
    "base_action_score": "meta_ranking",
    "base_severity_score": "meta_ranking",
    "evidence_strength": "meta_ranking",
    "missing_penalty": "meta_ranking",
    "contraindication_penalty": "meta_ranking",
    "data_completeness_score": "meta_ranking",
    "LBXGH": "clinical",
    "SBP_mean": "clinical",
    "DBP_mean": "clinical",
    "BMXBMI": "clinical",
    "BMXWAIST": "clinical",
    "HDL_low_flag": "clinical",
    "TC_high_flag": "clinical",
    "DM_severity_score": "clinical",
    "HTN_severity_score": "clinical",
    "Obesity_severity_score": "clinical",
    "Lifestyle_risk_score": "clinical",
    "Cardiometabolic_severity_score": "clinical",
    "DR1TSODI": "lifestyle",
    "DR1TFIBE": "lifestyle",
    "DR1TSUGR": "lifestyle",
    "Current_smoker_flag": "lifestyle",
    "Any_MVPA_flag": "lifestyle",
    "Alcohol_over_guideline_flag": "lifestyle",
    "SatFat_over_10pct_flag": "lifestyle",
    "Sodium_over_2300_flag": "lifestyle",
    "Sodium_over_1500_flag": "lifestyle",
    "Hypertensive_crisis_flag": "safety_context",
    "Severe_hyperglycemia_flag": "safety_context",
    "Severity_band": "safety_context",
    "bundle_family": "candidate_context",
    "severity": "candidate_context",
    "action_type": "candidate_context",
    "is_monitoring_bundle": "candidate_context",
    "is_diet_bundle": "candidate_context",
    "is_movement_bundle": "candidate_context",
    "is_weight_bundle": "candidate_context",
    "is_smoking_bundle": "candidate_context",
    "Hypertension": "clinical",
    "Diabetes": "clinical",
    "Prediabetes": "clinical",
}


def build_bundle_ml_xai_payload(
    bundle_row: dict,
    reranked_df: pd.DataFrame | None,
    safety_context_df: pd.DataFrame | None,
    participant_features_df: pd.DataFrame | None,
) -> dict:
    """
    Build SHAP-based local explanation for the representative candidate behind a bundle.

    Returns safe fallback payload if SHAP/artifacts/input rows are unavailable.
    """
    if shap is None:
        return _empty_xai_payload("shap_not_installed")

    if reranked_df is None or safety_context_df is None or participant_features_df is None:
        return _empty_xai_payload("missing_inputs")

    try:
        manifest = _load_manifest()
        model = joblib.load(manifest["model_path"])
        preprocessor = joblib.load(manifest["preprocessor_path"])

        training_meta = manifest.get("training", {})
        feature_columns = training_meta.get("feature_columns", [])
        numeric_columns = training_meta.get("numeric_columns", [])
        categorical_columns = training_meta.get("categorical_columns", [])

        representative_row = _select_representative_candidate(bundle_row, reranked_df)
        if representative_row is None:
            return _empty_xai_payload("representative_candidate_not_found")

        candidate_df = pd.DataFrame([representative_row])

        feature_df = build_ml_feature_table(
            safety_context_df=safety_context_df,
            ranked_df=candidate_df,
            participant_features_df=participant_features_df,
        )

        if feature_df.empty:
            return _empty_xai_payload("empty_feature_table")

        feature_df = _prepare_runtime_feature_df(
            feature_df=feature_df,
            feature_columns=feature_columns,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
        )

        X_proc = preprocessor.transform(feature_df)

        explainer = _build_tree_explainer(model)
        shap_values = explainer.shap_values(X_proc)

        local_values = _extract_positive_class_shap(shap_values)
        if local_values.ndim == 2:
            local_values = local_values[0]

        try:
            transformed_feature_names = list(preprocessor.get_feature_names_out())
        except Exception:
            transformed_feature_names = [f"f_{i}" for i in range(X_proc.shape[1])]

        local_series = pd.Series(local_values, index=transformed_feature_names, dtype="float64")

        top_positive_series = local_series.sort_values(ascending=False).head(5)
        top_negative_series = local_series.sort_values(ascending=True).head(5)

        top_positive = [
            _factor_dict(idx, val)
            for idx, val in top_positive_series.items()
            if float(val) > 0
        ]
        top_negative = [
            _factor_dict(idx, val)
            for idx, val in top_negative_series.items()
            if float(val) < 0
        ]

        preferred_groups = {"clinical", "lifestyle", "safety_context", "candidate_context"}

        clinical_positive = [
            item for item in top_positive
            if item["feature_group"] in preferred_groups
        ][:5]
        clinical_negative = [
            item for item in top_negative
            if item["feature_group"] in preferred_groups
        ][:5]

        if not clinical_positive:
            clinical_positive = top_positive[:3]

        if not clinical_negative:
            clinical_negative = top_negative[:3]

        clinician_summary, patient_summary = _build_xai_summaries(
            positive_factors=clinical_positive,
            negative_factors=clinical_negative,
        )

        return {
            "xai_status": "ok",
            "representative_action_code": representative_row.get("action_code"),
            "top_positive_factors": top_positive,
            "top_negative_factors": top_negative,
            "top_positive_factors_clinical_only": clinical_positive,
            "top_negative_factors_clinical_only": clinical_negative,
            "clinician_ml_xai_summary": clinician_summary,
            "patient_ml_xai_summary": patient_summary,
        }
    except Exception as exc:
        return _empty_xai_payload(f"xai_failure:{type(exc).__name__}")


def build_ml_xai_texts(xai_payload: dict) -> dict:
    if xai_payload.get("xai_status") != "ok":
        return {
            "clinician_ml_xai_explanation": "Local ML feature-attribution explanation was unavailable for this bundle.",
            "patient_ml_xai_explanation": "A detailed model-based explanation was not available for this recommendation.",
        }

    clinician_text = xai_payload.get("clinician_ml_xai_summary")
    patient_text = xai_payload.get("patient_ml_xai_summary")

    if clinician_text and patient_text:
        return {
            "clinician_ml_xai_explanation": clinician_text,
            "patient_ml_xai_explanation": patient_text,
        }

    pos = xai_payload.get("top_positive_factors", [])[:2]
    neg = xai_payload.get("top_negative_factors", [])[:2]

    if not pos and not neg:
        return {
            "clinician_ml_xai_explanation": "Local ML explanation was computed, but no strong feature-attribution pattern was identified.",
            "patient_ml_xai_explanation": "The model did not identify any especially strong additional factors for this recommendation.",
        }

    pos_text = ", ".join([str(item.get("feature_label", item.get("feature"))) for item in pos]) if pos else "none"
    neg_text = ", ".join([str(item.get("feature_label", item.get("feature"))) for item in neg]) if neg else "none"

    return {
        "clinician_ml_xai_explanation": (
            f"Top positive SHAP contributors: {pos_text}. "
            f"Top negative SHAP contributors: {neg_text}."
        ),
        "patient_ml_xai_explanation": (
            f"The model gave extra emphasis mainly because of: {pos_text}."
            if pos
            else "The model did not identify especially strong factors increasing this recommendation."
        ),
    }


def _select_representative_candidate(bundle_row: dict, reranked_df: pd.DataFrame) -> dict | None:
    participant_key = bundle_row.get("ParticipantKey")
    action_codes = bundle_row.get("action_codes", [])

    # allow fallback when the caller already passes a representative action row
    if not isinstance(action_codes, list) or len(action_codes) == 0:
        if bundle_row.get("action_code") is not None:
            return dict(bundle_row)
        return None

    subset = reranked_df.loc[
        (reranked_df["ParticipantKey"].astype(str) == str(participant_key))
        & (reranked_df["action_code"].astype(str).isin([str(a) for a in action_codes]))
    ].copy()

    if subset.empty:
        return None

    score_col = "final_score" if "final_score" in subset.columns else "total_score"
    subset = subset.sort_values(
        by=[score_col, "total_score", "confidence_score", "priority"],
        ascending=[False, False, False, False],
        kind="stable",
    )

    return subset.iloc[0].to_dict()


def _prepare_runtime_feature_df(
    feature_df: pd.DataFrame,
    feature_columns: list[str],
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> pd.DataFrame:
    out = feature_df.copy()

    for col in feature_columns:
        if col not in out.columns:
            if col in numeric_columns:
                out[col] = np.nan
            else:
                out[col] = None

    out = out[feature_columns].copy()

    for col in numeric_columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in categorical_columns:
        if col in out.columns:
            out[col] = out[col].astype("object")
            out[col] = out[col].where(~pd.isna(out[col]), None)

    return out


def _build_tree_explainer(model: Any):
    if hasattr(model, "booster_"):
        return shap.TreeExplainer(model.booster_)
    return shap.TreeExplainer(model)


def _extract_positive_class_shap(shap_values: Any) -> np.ndarray:
    if isinstance(shap_values, list):
        if len(shap_values) >= 2:
            return np.asarray(shap_values[1])
        return np.asarray(shap_values[0])
    return np.asarray(shap_values)


def _feature_label(feature_name: str) -> str:
    if feature_name in FEATURE_LABEL_MAP:
        return FEATURE_LABEL_MAP[feature_name]

    raw = feature_name
    if "__" in raw:
        raw = raw.split("__", 1)[1]

    return raw.replace("_", " ")


def _feature_group(feature_name: str) -> str:
    raw = feature_name
    if "__" in raw:
        raw = raw.split("__", 1)[1]

    if raw in FEATURE_GROUP_MAP:
        return FEATURE_GROUP_MAP[raw]

    for key, group in FEATURE_GROUP_MAP.items():
        if raw.startswith(key):
            return group

    return "other"


def _factor_dict(feature_name: str, shap_value: float) -> dict:
    return {
        "feature": str(feature_name),
        "feature_label": _feature_label(str(feature_name)),
        "feature_group": _feature_group(str(feature_name)),
        "shap_value": float(shap_value),
    }


def _build_xai_summaries(
    positive_factors: list[dict],
    negative_factors: list[dict],
) -> tuple[str, str]:
    pos_labels = [x.get("feature_label") for x in positive_factors[:3] if x.get("feature_label")]
    neg_labels = [x.get("feature_label") for x in negative_factors[:3] if x.get("feature_label")]

    if pos_labels and neg_labels:
        clinician = (
            f"Features increasing relative ML emphasis included: {', '.join(pos_labels)}. "
            f"Features decreasing relative ML emphasis included: {', '.join(neg_labels)}."
        )
        patient = (
            f"The model placed more emphasis on this recommendation because of factors such as {', '.join(pos_labels[:2]).lower()}. "
            f"It placed less additional emphasis because of factors such as {', '.join(neg_labels[:2]).lower()}."
        )
        return clinician, patient

    if pos_labels:
        clinician = f"Features increasing relative ML emphasis included: {', '.join(pos_labels)}."
        patient = (
            f"The model placed more emphasis on this recommendation because of factors such as "
            f"{', '.join(pos_labels[:2]).lower()}."
        )
        return clinician, patient

    if neg_labels:
        clinician = f"Features decreasing relative ML emphasis included: {', '.join(neg_labels)}."
        patient = (
            f"The model placed less additional emphasis because of factors such as {', '.join(neg_labels[:2]).lower()}."
        )
        return clinician, patient

    return (
        "Local ML explanation was computed, but no strong feature-attribution pattern was identified.",
        "The model did not identify especially strong additional factors for this recommendation.",
    )


def _load_manifest() -> dict:
    import json

    with open(ACTIVE_MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _empty_xai_payload(reason: str) -> dict:
    return {
        "xai_status": reason,
        "representative_action_code": None,
        "top_positive_factors": [],
        "top_negative_factors": [],
        "top_positive_factors_clinical_only": [],
        "top_negative_factors_clinical_only": [],
        "clinician_ml_xai_summary": "Local ML feature-attribution explanation was unavailable for this bundle.",
        "patient_ml_xai_summary": "A detailed model-based explanation was not available for this recommendation.",
    }
