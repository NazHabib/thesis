from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from django.conf import settings

from apps.ml_layer.services.ml_reranking.feature_builder import build_ml_feature_table

try:
    import shap
except Exception:  # pragma: no cover
    shap = None


ACTIVE_MANIFEST_PATH = (
    Path(settings.BASE_DIR)
    / "apps"
    / "ml_layer"
    / "artifacts"
    / "manifests"
    / "active_manifest.json"
)


def run_participant_shap_explanation(
    participant_key: str,
    safety_context_df: pd.DataFrame,
    reranked_df: pd.DataFrame,
    participant_features_df: pd.DataFrame,
) -> dict:
    if shap is None:
        raise RuntimeError("shap is not installed in the current environment.")

    participant_key = str(participant_key)

    safety_one = safety_context_df.loc[
        safety_context_df["ParticipantKey"].astype(str) == participant_key
    ].copy()

    reranked_one = reranked_df.loc[
        reranked_df["ParticipantKey"].astype(str) == participant_key
    ].copy()

    features_one = participant_features_df.loc[
        participant_features_df["ParticipantKey"].astype(str) == participant_key
    ].copy()

    if safety_one.empty:
        raise ValueError(f"No safety context found for participant: {participant_key}")
    if reranked_one.empty:
        raise ValueError(f"No reranked recommendations found for participant: {participant_key}")
    if features_one.empty:
        raise ValueError(f"No participant features found for participant: {participant_key}")

    manifest = _load_manifest()
    model = joblib.load(manifest["model_path"])
    preprocessor = joblib.load(manifest["preprocessor_path"])

    feature_columns = manifest.get("training", {}).get("feature_columns", [])
    numeric_columns = manifest.get("training", {}).get("numeric_columns", [])
    categorical_columns = manifest.get("training", {}).get("categorical_columns", [])

    if not feature_columns:
        raise ValueError("Active ML manifest does not contain training.feature_columns")

    feature_df = build_ml_feature_table(
        safety_context_df=safety_one,
        ranked_df=reranked_one,
        participant_features_df=features_one,
    )

    if feature_df.empty:
        raise ValueError(f"Feature table is empty for participant: {participant_key}")

    raw_feature_snapshot_cols = [c for c in feature_columns if c in feature_df.columns]

    feature_input_df = _prepare_runtime_feature_input(
        feature_df=feature_df,
        feature_columns=feature_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )

    X_proc = preprocessor.transform(feature_input_df)

    explainer = _build_tree_explainer(model)
    shap_values = explainer.shap_values(X_proc)
    shap_matrix = _extract_positive_class_shap(shap_values)

    if shap_matrix.ndim == 1:
        shap_matrix = shap_matrix.reshape(1, -1)

    try:
        transformed_feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        transformed_feature_names = [f"f_{i}" for i in range(shap_matrix.shape[1])]

    joined = reranked_one.copy().reset_index(drop=True)

    per_action_records = []

    for i in range(len(joined)):
        row_series = pd.Series(shap_matrix[i], index=transformed_feature_names, dtype="float64")

        pos = row_series.sort_values(ascending=False).head(5)
        neg = row_series.sort_values(ascending=True).head(5)

        top_pos_list = [
            {"feature": str(idx), "shap_value": float(val)}
            for idx, val in pos.items()
            if float(val) > 0
        ]
        top_neg_list = [
            {"feature": str(idx), "shap_value": float(val)}
            for idx, val in neg.items()
            if float(val) < 0
        ]

        per_action_records.append({
            "ParticipantKey": participant_key,
            "action_code": joined.iloc[i]["action_code"],
            "bundle_family": joined.iloc[i].get("bundle_family"),
            "total_score": float(joined.iloc[i].get("total_score", 0.0)),
            "final_score": float(joined.iloc[i].get("final_score", joined.iloc[i].get("total_score", 0.0))),
            "ml_score": float(joined.iloc[i].get("ml_score", 0.0)),
            "lambda_u": float(joined.iloc[i].get("lambda_u", 0.0)),
            "ml_contrib": float(joined.iloc[i].get("ml_contrib", 0.0)),
            "rerank_status": joined.iloc[i].get("rerank_status"),
            "top_positive_factors": top_pos_list,
            "top_negative_factors": top_neg_list,
        })

    global_mean_abs = np.abs(shap_matrix).mean(axis=0)
    global_importance_df = pd.DataFrame({
        "feature": transformed_feature_names,
        "mean_abs_shap": global_mean_abs,
    }).sort_values("mean_abs_shap", ascending=False, kind="stable")

    representative_cols = [
        c for c in ["action_code", "bundle_family", "total_score", "final_score", "ml_score", "lambda_u", "ml_contrib"]
        if c in joined.columns
    ]
    candidate_summary = joined[representative_cols].copy().to_dict(orient="records")

    feature_snapshot = feature_df[["ParticipantKey"] + raw_feature_snapshot_cols].head(1).to_dict(orient="records")

    return {
        "participant_key": participant_key,
        "model_version": manifest.get("model_version"),
        "feature_schema_version": manifest.get("feature_schema_version"),
        "calibration_version": manifest.get("calibration_version"),
        "candidate_summary": candidate_summary,
        "feature_snapshot": feature_snapshot,
        "global_feature_importance": global_importance_df.to_dict(orient="records"),
        "per_action_explanations": per_action_records,
    }


def save_participant_shap_outputs(result: dict, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    participant_key = str(result["participant_key"]).replace("/", "_")
    json_path = output_dir / f"{participant_key}_shap_report.json"
    csv_path = output_dir / f"{participant_key}_shap_per_action.csv"
    global_csv_path = output_dir / f"{participant_key}_shap_global_importance.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    per_action_df = pd.DataFrame(result["per_action_explanations"])
    per_action_df.to_csv(csv_path, index=False)

    global_df = pd.DataFrame(result["global_feature_importance"])
    global_df.to_csv(global_csv_path, index=False)

    return {
        "json_path": json_path,
        "csv_path": csv_path,
        "global_csv_path": global_csv_path,
    }


def _prepare_runtime_feature_input(
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


def _build_tree_explainer(model):
    if hasattr(model, "booster_"):
        return shap.TreeExplainer(model.booster_)
    return shap.TreeExplainer(model)


def _extract_positive_class_shap(shap_values):
    if isinstance(shap_values, list):
        if len(shap_values) >= 2:
            return np.asarray(shap_values[1])
        return np.asarray(shap_values[0])
    return np.asarray(shap_values)


def _load_manifest() -> dict:
    with open(ACTIVE_MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
