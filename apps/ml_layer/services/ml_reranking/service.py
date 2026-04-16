import pandas as pd
import joblib

from .artifact_loader import MLArtifactLoader
from .config import (
    ACTIVE_MANIFEST_PATH,
    DEFAULT_CALIBRATION_VERSION,
    DEFAULT_MAX_ML_CONTRIB,
    DEFAULT_MODEL_VERSION,
    DEFAULT_SCHEMA_VERSION,
    LOW_SCORE_SPREAD_THRESHOLD,
    MIN_REQUIRED_CANDIDATES_FOR_ML,
    ML_RERANKING_VERSION,
)
from .contracts import (
    FINAL_RERANKED_COLUMNS,
    validate_ranked_columns,
    validate_safety_context_columns,
    validate_participant_features_columns,
)
from .fallback import rules_only_fallback
from .feature_builder import build_ml_feature_table
from .lambda_policy import compute_lambda
from .schema_validator import validate_ml_feature_table


def run_ml_reranking_layer(
    safety_context_df: pd.DataFrame,
    ranked_df: pd.DataFrame,
    participant_features_df: pd.DataFrame | None = None,
) -> dict:
    """
    Post-ranking, pre-bundling reranking layer.

    Safety rules:
    - never introduces new candidates
    - never removes safety filtering
    - disabled in triage
    - bounded ML influence via lambda and clipping
    - falls back to rules-only on any artifact/data issue
    """
    validate_safety_context_columns(safety_context_df)
    validate_ranked_columns(ranked_df)

    if participant_features_df is not None:
        validate_participant_features_columns(participant_features_df)

    if ranked_df.empty:
        return {
            "reranked_recommendations": ranked_df.copy(),
            "manifest": _build_manifest(
                safety_context_df=safety_context_df,
                input_ranked_df=ranked_df,
                output_df=ranked_df.copy(),
                rerank_mode="empty_input",
            ),
        }

    merged_df = build_ml_feature_table(
        safety_context_df=safety_context_df,
        ranked_df=ranked_df,
        participant_features_df=participant_features_df,
    )

    valid, issues = validate_ml_feature_table(merged_df)
    if not valid:
        reranked_df = rules_only_fallback(
            ranked_df=ranked_df,
            reason=f"schema_mismatch:{'; '.join(issues)}",
        )
        reranked_df = reranked_df[FINAL_RERANKED_COLUMNS].copy()

        return {
            "reranked_recommendations": reranked_df,
            "manifest": _build_manifest(
                safety_context_df=safety_context_df,
                input_ranked_df=ranked_df,
                output_df=reranked_df,
                rerank_mode="rules_only_fallback",
            ),
        }

    artifacts = _load_runtime_artifacts()

    if not artifacts["artifacts_ready"]:
        reranked_df = rules_only_fallback(
            ranked_df=ranked_df,
            reason="artifact_load_failure",
            model_version=artifacts["model_version"],
            schema_version=artifacts["schema_version"],
            calibration_version=artifacts["calibration_version"],
        )
        reranked_df = reranked_df[FINAL_RERANKED_COLUMNS].copy()

        return {
            "reranked_recommendations": reranked_df,
            "manifest": _build_manifest(
                safety_context_df=safety_context_df,
                input_ranked_df=ranked_df,
                output_df=reranked_df,
                rerank_mode="rules_only_fallback",
            ),
        }

    reranked_df = _apply_patient_level_reranking(
        merged_df=merged_df,
        artifacts=artifacts,
    )

    reranked_df = reranked_df[FINAL_RERANKED_COLUMNS].copy()

    return {
        "reranked_recommendations": reranked_df,
        "manifest": _build_manifest(
            safety_context_df=safety_context_df,
            input_ranked_df=ranked_df,
            output_df=reranked_df,
            rerank_mode=_infer_rerank_mode(reranked_df),
        ),
    }


def _apply_patient_level_reranking(
    merged_df: pd.DataFrame,
    artifacts: dict,
) -> pd.DataFrame:
    out_frames: list[pd.DataFrame] = []

    for participant_key, group_df in merged_df.groupby("ParticipantKey", sort=False):
        participant_out = _rerank_single_participant(
            participant_key=participant_key,
            df=group_df.copy(),
            artifacts=artifacts,
        )
        out_frames.append(participant_out)

    if not out_frames:
        return merged_df.copy()

    return pd.concat(out_frames, ignore_index=True)


def _rerank_single_participant(
    participant_key: str,
    df: pd.DataFrame,
    artifacts: dict,
) -> pd.DataFrame:
    triage_mode = bool(df["triage_mode"].fillna(False).iloc[0])

    if triage_mode:
        return rules_only_fallback(
            ranked_df=df,
            reason="triage_mode",
            model_version=artifacts["model_version"],
            schema_version=artifacts["schema_version"],
            calibration_version=artifacts["calibration_version"],
        )

    if len(df) < MIN_REQUIRED_CANDIDATES_FOR_ML:
        return rules_only_fallback(
            ranked_df=df,
            reason="insufficient_candidates",
            model_version=artifacts["model_version"],
            schema_version=artifacts["schema_version"],
            calibration_version=artifacts["calibration_version"],
        )

    feature_tier = df["feature_tier"].iloc[0] if "feature_tier" in df.columns else "minimal"
    data_completeness_score = (
        df["data_completeness_score"].iloc[0]
        if "data_completeness_score" in df.columns
        else None
    )
    max_rule_confidence_score = float(df["confidence_score"].fillna(0.0).max())

    lambda_u = compute_lambda(
        triage_mode=triage_mode,
        feature_tier=feature_tier,
        data_completeness_score=data_completeness_score,
        max_rule_confidence_score=max_rule_confidence_score,
    )

    try:
        feature_df = _select_runtime_feature_columns(
            df=df,
            expected_columns=artifacts["feature_columns"],
            numeric_columns=artifacts.get("numeric_columns", []),
            categorical_columns=artifacts.get("categorical_columns", []),
        )
        X_proc = artifacts["preprocessor"].transform(feature_df)
        pred_raw = artifacts["model"].predict_proba(X_proc)[:, 1]
        pred_cal = artifacts["calibrator"].predict(pred_raw)
        pred_cal = pd.Series(pred_cal, index=df.index).clip(lower=0.0, upper=1.0)
    except Exception as exc:
        return rules_only_fallback(
            ranked_df=df,
            reason=f"prediction_failure:{type(exc).__name__}",
            model_version=artifacts["model_version"],
            schema_version=artifacts["schema_version"],
            calibration_version=artifacts["calibration_version"],
        )

    score_spread = float(pred_cal.max() - pred_cal.min()) if len(pred_cal) else 0.0
    if score_spread < LOW_SCORE_SPREAD_THRESHOLD:
        return rules_only_fallback(
            ranked_df=df,
            reason="low_ml_score_dispersion",
            model_version=artifacts["model_version"],
            schema_version=artifacts["schema_version"],
            calibration_version=artifacts["calibration_version"],
        )

    out = df.copy()
    out["ml_score"] = pred_cal.astype(float)
    out["lambda_u"] = float(lambda_u)
    out["ml_contrib"] = (out["lambda_u"] * out["ml_score"]).clip(
        lower=0.0,
        upper=artifacts["max_ml_contrib"],
    )

    # Experiment E: normalize rule score within participant before fusion
    total_min = float(out["total_score"].min())
    total_max = float(out["total_score"].max())
    score_range = total_max - total_min

    if score_range <= 1e-8:
        out["final_score"] = out["ml_contrib"]
    else:
        out["final_score"] = (
            (out["total_score"] - total_min) / score_range
        ) + out["ml_contrib"]

    out["rerank_status"] = "ml_reranked"
    out["fallback_reason"] = None
    out["ml_model_version"] = artifacts["model_version"]
    out["ml_schema_version"] = artifacts["schema_version"]
    out["ml_calibration_version"] = artifacts["calibration_version"]

    out = out.sort_values(
        by=["final_score", "total_score", "confidence_score", "priority"],
        ascending=[False, False, False, False],
        kind="stable",
    ).reset_index(drop=True)

    return out


def _select_runtime_feature_columns(
    df: pd.DataFrame,
    expected_columns: list[str],
    numeric_columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
) -> pd.DataFrame:
    out = df.copy()

    numeric_columns = numeric_columns or []
    categorical_columns = categorical_columns or []

    for col in expected_columns:
        if col not in out.columns:
            if col in numeric_columns:
                out[col] = float("nan")
            else:
                out[col] = None

    out = out[expected_columns].copy()

    for col in numeric_columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in categorical_columns:
        if col in out.columns:
            out[col] = out[col].astype("object")
            out[col] = out[col].where(~pd.isna(out[col]), None)

    return out


def _load_runtime_artifacts() -> dict:
    metadata = MLArtifactLoader.build_metadata(ACTIVE_MANIFEST_PATH)
    metadata.setdefault("model_version", DEFAULT_MODEL_VERSION)
    metadata.setdefault("schema_version", DEFAULT_SCHEMA_VERSION)
    metadata.setdefault("calibration_version", DEFAULT_CALIBRATION_VERSION)
    metadata.setdefault("max_ml_contrib", DEFAULT_MAX_ML_CONTRIB)

    if not metadata["artifacts_ready"]:
        metadata["model"] = None
        metadata["preprocessor"] = None
        metadata["calibrator"] = None
        metadata["feature_columns"] = []
        return metadata

    try:
        manifest = metadata["manifest"]
        model = joblib.load(manifest["model_path"])
        preprocessor = joblib.load(manifest["preprocessor_path"])
        calibrator = joblib.load(manifest["calibrator_path"])

        training_meta = manifest.get("training", {})
        feature_columns = training_meta.get("feature_columns", [])
        numeric_columns = training_meta.get("numeric_columns", [])
        categorical_columns = training_meta.get("categorical_columns", [])

        metadata["model"] = model
        metadata["preprocessor"] = preprocessor
        metadata["calibrator"] = calibrator
        metadata["feature_columns"] = feature_columns
        metadata["numeric_columns"] = numeric_columns
        metadata["categorical_columns"] = categorical_columns
        metadata["artifacts_ready"] = True
        return metadata
    except Exception:
        metadata["model"] = None
        metadata["preprocessor"] = None
        metadata["calibrator"] = None
        metadata["feature_columns"] = []
        metadata["numeric_columns"] = []
        metadata["categorical_columns"] = []
        metadata["artifacts_ready"] = False
        return metadata


def _infer_rerank_mode(df: pd.DataFrame) -> str:
    if df.empty or "rerank_status" not in df.columns:
        return "unknown"

    statuses = set(df["rerank_status"].dropna().astype(str).unique().tolist())

    if statuses == {"ml_reranked"}:
        return "ml_reranked"
    if statuses == {"rules_only_fallback"}:
        return "rules_only_fallback"
    return "mixed"


def _build_manifest(
    safety_context_df: pd.DataFrame,
    input_ranked_df: pd.DataFrame,
    output_df: pd.DataFrame,
    rerank_mode: str,
) -> dict:
    n_participants = int(safety_context_df["ParticipantKey"].nunique())

    fallback_counts = {}
    if "fallback_reason" in output_df.columns:
        fallback_counts = {
            str(k): int(v)
            for k, v in output_df["fallback_reason"].fillna("none").value_counts(dropna=False).to_dict().items()
        }

    rerank_status_counts = {}
    if "rerank_status" in output_df.columns:
        rerank_status_counts = {
            str(k): int(v)
            for k, v in output_df["rerank_status"].fillna("unknown").value_counts(dropna=False).to_dict().items()
        }

    return {
        "ml_reranking_version": ML_RERANKING_VERSION,
        "fusion_strategy": "participant_minmax_total_score_plus_ml_contrib",
        "participants": n_participants,
        "input_ranked_candidates": int(len(input_ranked_df)),
        "output_reranked_candidates": int(len(output_df)),
        "triage_count": int(safety_context_df["triage_mode"].fillna(False).sum()),
        "feature_tier_counts": {
            "minimal": int((safety_context_df["feature_tier"] == "minimal").sum()),
            "core": int((safety_context_df["feature_tier"] == "core").sum()),
            "full": int((safety_context_df["feature_tier"] == "full").sum()),
        },
        "rerank_mode": rerank_mode,
        "rerank_status_counts": rerank_status_counts,
        "fallback_reason_counts": fallback_counts,
        "score_summary": {
            "min_total_score": float(output_df["total_score"].min()) if not output_df.empty else 0.0,
            "max_total_score": float(output_df["total_score"].max()) if not output_df.empty else 0.0,
            "mean_total_score": round(float(output_df["total_score"].mean()), 4) if not output_df.empty else 0.0,
            "min_final_score": float(output_df["final_score"].min()) if ("final_score" in output_df.columns and not output_df.empty) else 0.0,
            "max_final_score": float(output_df["final_score"].max()) if ("final_score" in output_df.columns and not output_df.empty) else 0.0,
            "mean_final_score": round(float(output_df["final_score"].mean()), 4) if ("final_score" in output_df.columns and not output_df.empty) else 0.0,
        },
    }
