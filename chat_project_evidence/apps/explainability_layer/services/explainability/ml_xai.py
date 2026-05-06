from __future__ import annotations

from pathlib import Path
from typing import Any

from .feature_labels import humanize_feature_name, feature_group, is_clinical_only_feature

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

        preferred_groups = {"clinical", "lifestyle", "safety_context", "candidate_context", "temporal"}

        clinical_positive = [
            item for item in top_positive
            if item["feature_group"] in preferred_groups
        ][:5]
        clinical_negative = [
            item for item in top_negative
            if item["feature_group"] in preferred_groups
        ][:5]
        
        clinical_positive = _deduplicate_factors(clinical_positive)
        clinical_negative = _deduplicate_factors(clinical_negative)

        clinical_positive = _drop_contradictory_priority_labels(clinical_positive)
        clinical_negative = _drop_contradictory_priority_labels(clinical_negative)

        if not clinical_positive:
            clinical_positive = top_positive[:3]

        if not clinical_negative:
            clinical_negative = top_negative[:3]
            
        patient_positive = [item for item in clinical_positive if _is_patient_safe_factor(item)][:3]
        patient_negative = [item for item in clinical_negative if _is_patient_safe_factor(item)][:3]

        clinician_summary, patient_summary = _build_xai_summaries(
            positive_factors=clinical_positive,
            negative_factors=clinical_negative,
            patient_positive_factors=patient_positive,
            patient_negative_factors=patient_negative,
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


def _factor_dict(feature_name: str, shap_value: float) -> dict:
    return {
        "feature": str(feature_name),
        "feature_label": humanize_feature_name(str(feature_name)),
        "feature_group": feature_group(str(feature_name)),
        "shap_value": float(shap_value),
    }

def _clean_factor_label(label: str) -> str:
    text = str(label or "").strip()
    if not text:
        return ""

    replacements = {
        "hba1c": "HbA1c",
        "day-1": "Day-1",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def _is_patient_safe_factor(item: dict) -> bool:
    group = str(item.get("feature_group", ""))
    label = str(item.get("feature_label", "")).lower()

    blocked_substrings = [
        "action code",
        "confidence label",
        "upstream confidence",
        "recommendation context",
        "supporting informational context",
        "direct recommendation context",
        "escalation context",
        "data completeness score",
        "rule-based total score",
        "normalized rule score",
        "base severity score",
        "base action score",
        "domain alignment score",
        "evidence strength score",
        "missing-data penalty",
        "contraindication penalty",
	"higher clinical priority",
	"lower clinical priority",
	"high candidate severity",
	"low candidate severity",
	"moderate candidate severity",
    ]

    if group == "meta_ranking":
        return False

    return not any(token in label for token in blocked_substrings)


def _deduplicate_factors(items: list[dict]) -> list[dict]:
    seen = set()
    out = []

    for item in items:
        label = _clean_factor_label(item.get("feature_label", ""))
        group = str(item.get("feature_group", ""))
        key = (label.lower(), group)
        if label and key not in seen:
            seen.add(key)
            item = item.copy()
            item["feature_label"] = label
            out.append(item)

    return out


def _drop_contradictory_priority_labels(items: list[dict]) -> list[dict]:
    labels = {str(item.get("feature_label", "")).lower() for item in items}

    high_tokens = {
        "higher clinical priority",
        "high candidate severity",
        "higher-priority recommendation context",
    }
    low_tokens = {
        "lower clinical priority",
        "low candidate severity",
        "lower-priority recommendation context",
    }

    has_high = any(label in high_tokens for label in labels)
    has_low = any(label in low_tokens for label in labels)

    if has_high and has_low:
        items = [
            item for item in items
            if str(item.get("feature_label", "")).lower() not in (high_tokens | low_tokens)
        ]

    return items


def _join_labels(labels: list[str]) -> str:
    labels = [str(x).strip() for x in labels if str(x).strip()]
    if not labels:
        return "none"
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return f"{', '.join(labels[:-1])}, and {labels[-1]}"

def _build_xai_summaries(
    positive_factors: list[dict],
    negative_factors: list[dict],
    patient_positive_factors: list[dict] | None = None,
    patient_negative_factors: list[dict] | None = None,
) -> tuple[str, str]:
    positive_factors = _deduplicate_factors(positive_factors)
    negative_factors = _deduplicate_factors(negative_factors)

    positive_factors = _drop_contradictory_priority_labels(positive_factors)
    negative_factors = _drop_contradictory_priority_labels(negative_factors)

    pos_labels = [x.get("feature_label") for x in positive_factors[:3] if x.get("feature_label")]
    neg_labels = [x.get("feature_label") for x in negative_factors[:3] if x.get("feature_label")]

    patient_positive_factors = patient_positive_factors or []
    patient_negative_factors = patient_negative_factors or []

    patient_pos_labels = [
        x.get("feature_label") for x in patient_positive_factors[:2] if x.get("feature_label")
    ]
    patient_neg_labels = [
        x.get("feature_label") for x in patient_negative_factors[:2] if x.get("feature_label")
    ]

    if pos_labels and neg_labels:
        clinician = (
            f"Factors increasing relative ML emphasis included {_join_labels(pos_labels)}. "
            f"Factors decreasing relative ML emphasis included {_join_labels(neg_labels)}."
        )
    elif pos_labels:
        clinician = f"Factors increasing relative ML emphasis included {_join_labels(pos_labels)}."
    elif neg_labels:
        clinician = f"Factors decreasing relative ML emphasis included {_join_labels(neg_labels)}."
    else:
        clinician = "Local ML explanation was computed, but no especially strong factor pattern was identified."

    if patient_pos_labels and patient_neg_labels:
        patient = (
            f"The model gave relatively more weight to this recommendation because of factors such as {_join_labels(patient_pos_labels)}. "
            f"Factors such as {_join_labels(patient_neg_labels)} reduced that additional emphasis."
        )
    elif patient_pos_labels:
        patient = (
            f"The model gave relatively more weight to this recommendation because of factors such as "
            f"{_join_labels(patient_pos_labels)}."
        )
    elif patient_neg_labels:
        patient = (
            f"Factors such as {_join_labels(patient_neg_labels)} reduced the model's additional emphasis."
        )
    else:
        patient = (
            "The model did not identify especially strong additional health or lifestyle factors for this recommendation."
        )

    return clinician, patient


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
