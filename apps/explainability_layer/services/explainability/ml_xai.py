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
        feature_columns = manifest.get("training", {}).get("feature_columns", [])

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

        for col in feature_columns:
            if col not in feature_df.columns:
                feature_df[col] = pd.NA

        feature_df = feature_df[feature_columns].copy()

        X_proc = preprocessor.transform(feature_df)

        explainer = _build_tree_explainer(model)
        shap_values = explainer.shap_values(X_proc)

        # Binary classification compatibility:
        # shap can return list[class0, class1] or array
        local_values = _extract_positive_class_shap(shap_values)

        if local_values.ndim == 2:
            local_values = local_values[0]

        try:
            transformed_feature_names = list(preprocessor.get_feature_names_out())
        except Exception:
            transformed_feature_names = [f"f_{i}" for i in range(X_proc.shape[1])]

        local_series = pd.Series(local_values, index=transformed_feature_names, dtype="float64")

        top_positive = (
            local_series.sort_values(ascending=False)
            .head(3)
        )
        top_negative = (
            local_series.sort_values(ascending=True)
            .head(3)
        )

        return {
            "xai_status": "ok",
            "representative_action_code": representative_row.get("action_code"),
            "top_positive_factors": [
                {"feature": str(idx), "shap_value": float(val)}
                for idx, val in top_positive.items()
                if float(val) > 0
            ],
            "top_negative_factors": [
                {"feature": str(idx), "shap_value": float(val)}
                for idx, val in top_negative.items()
                if float(val) < 0
            ],
        }
    except Exception as exc:
        return _empty_xai_payload(f"xai_failure:{type(exc).__name__}")


def build_ml_xai_texts(xai_payload: dict) -> dict:
    if xai_payload.get("xai_status") != "ok":
        return {
            "clinician_ml_xai_explanation": "Local ML feature-attribution explanation was unavailable for this bundle.",
            "patient_ml_xai_explanation": "A detailed model-based explanation was not available for this recommendation.",
        }

    pos = xai_payload.get("top_positive_factors", [])[:2]
    neg = xai_payload.get("top_negative_factors", [])[:2]

    if not pos and not neg:
        return {
            "clinician_ml_xai_explanation": "Local ML explanation was computed, but no strong feature-attribution pattern was identified.",
            "patient_ml_xai_explanation": "The model did not identify any especially strong additional factors for this recommendation.",
        }

    pos_text = ", ".join([str(item["feature"]) for item in pos]) if pos else "none"
    neg_text = ", ".join([str(item["feature"]) for item in neg]) if neg else "none"

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

    if not isinstance(action_codes, list) or len(action_codes) == 0:
        return None

    subset = reranked_df.loc[
        (reranked_df["ParticipantKey"] == participant_key)
        & (reranked_df["action_code"].isin(action_codes))
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
    }
