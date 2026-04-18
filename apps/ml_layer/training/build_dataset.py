from __future__ import annotations

import pandas as pd

from apps.ml_layer.services.ml_reranking.feature_builder import add_interaction_features

from apps.common.clinical_safety.temporal import build_temporal_features


TRAINING_REQUIRED_SAFETY_COLUMNS = [
    "ParticipantKey",
    "feature_tier",
    "data_completeness_score",
    "triage_mode",
    "confidence_ceiling_global",
]

TRAINING_REQUIRED_RANKED_COLUMNS = [
    "ParticipantKey",
    "action_code",
    "action_type",
    "severity",
    "priority",
    "tags",
    "evidence",
    "source_rule_ids",
    "bundle_family",
    "coldstart_note_codes",
    "coldstart_confidence_ceiling",
    "base_severity_score",
    "base_action_score",
    "evidence_strength",
    "domain_alignment",
    "missing_penalty",
    "contraindication_penalty",
    "total_score",
    "score_norm",
    "confidence_score",
    "confidence_label",
]

DEFAULT_PARTICIPANT_FEATURE_COLUMNS = [
    "ParticipantKey",
    "RIDAGEYR",
    "RIAGENDR",
    "BMXBMI",
    "BMXWAIST",
    "LBXGH",
    "SBP_mean",
    "DBP_mean",
    "Any_MVPA_flag",
    "Current_smoker_flag",
    "DietDay1_reliable_flag",
    "DR1TSODI",
    "DR1TSUGR",
    "DR1TFIBE",
    "Alcohol_over_guideline_flag",
    "Hypertension",
    "Hypertensive_crisis_flag",
    "Diabetes",
    "Prediabetes",
    "Severe_hyperglycemia_flag",
    "Severe_obesity_flag",
    "Sedentary_high_flag",
    "Sodium_over_2300_flag",
    "Sodium_over_1500_flag",
    "SatFat_over_10pct_flag",
    "HDL_low_flag",
    "TC_high_flag",
    "Comorbidity",
    "HTN_severity_score",
    "DM_severity_score",
    "Obesity_severity_score",
    "Lifestyle_risk_score",
    "Cardiometabolic_severity_score",
    "Severity_band",
    "followup_urgency",
    "followup_horizon_days",
    "monitoring_need_level",
    "temporal_risk_tier",
    "persistence_risk_proxy",
    "repeat_measurement_available",
    "temporal_data_quality",
    "longitudinal_ready_flag",
]

DEFAULT_LABEL_SCHEME = "rule_top_k"
TOP_K_POSITIVE_DEFAULT = 3
MIN_SCORE_NORM_DEFAULT = 0.80


def build_ml_training_dataset(
    safety_context_df: pd.DataFrame,
    ranked_df: pd.DataFrame,
    participant_features_df: pd.DataFrame | None = None,
    top_k_positive: int = TOP_K_POSITIVE_DEFAULT,
    exclude_triage: bool = True,
    label_scheme: str = DEFAULT_LABEL_SCHEME,
    min_score_norm: float = MIN_SCORE_NORM_DEFAULT,
) -> pd.DataFrame:
    """
    Build the supervised ML reranking table.

    Training unit:
        one row = (ParticipantKey, candidate recommendation)

    Supported label schemes:
    - rule_top_k
    - rule_top_k_with_min_score
    - rule_top_k_plus_bundle_retention
    """
    _validate_required_columns(
        safety_context_df,
        TRAINING_REQUIRED_SAFETY_COLUMNS,
        "safety_context_df",
    )
    _validate_required_columns(
        ranked_df,
        TRAINING_REQUIRED_RANKED_COLUMNS,
        "ranked_df",
    )

    df = ranked_df.copy()

    safety_cols = [c for c in TRAINING_REQUIRED_SAFETY_COLUMNS if c in safety_context_df.columns]
    df = df.merge(
        safety_context_df[safety_cols].drop_duplicates(subset=["ParticipantKey"]),
        on="ParticipantKey",
        how="left",
        validate="m:1",
    )

    if participant_features_df is not None and not participant_features_df.empty:
        participant_features_df = build_temporal_features(participant_features_df.copy())

        if "ParticipantKey" not in participant_features_df.columns:
            raise ValueError("participant_features_df must contain 'ParticipantKey'")

        feat_cols = [c for c in DEFAULT_PARTICIPANT_FEATURE_COLUMNS if c in participant_features_df.columns]
        participant_df = participant_features_df[feat_cols].drop_duplicates(subset=["ParticipantKey"]).copy()

        df = df.merge(
            participant_df,
            on="ParticipantKey",
            how="left",
            validate="m:1",
        )

    if exclude_triage and "triage_mode" in df.columns:
        df = df.loc[~df["triage_mode"].fillna(False)].copy()

    df = _deduplicate_participant_candidates(df)
    df = _add_rule_rank_columns(df)

    df = _apply_label_scheme(
        df=df,
        label_scheme=label_scheme,
        top_k_positive=top_k_positive,
        min_score_norm=min_score_norm,
    )

    df = add_interaction_features(df)

    df["label_source"] = label_scheme
    df["training_row_type"] = "participant_candidate"
    df["feature_schema_version"] = "v1"
    df["label_schema_version"] = "v2"

    sort_cols = ["ParticipantKey", "rule_rank", "total_score", "priority", "action_code"]
    sort_cols = [c for c in sort_cols if c in df.columns]
    df = df.sort_values(sort_cols, ascending=[True] * len(sort_cols), kind="stable").reset_index(drop=True)

    return df


def _validate_required_columns(df: pd.DataFrame, required: list[str], df_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} missing required columns: {missing}")


def _deduplicate_participant_candidates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    key_cols = ["ParticipantKey", "action_code"]
    duplicate_mask = df.duplicated(subset=key_cols, keep=False)

    if not duplicate_mask.any():
        return df.copy()

    agg_spec: dict[str, str] = {}
    for col in df.columns:
        if col in {"ParticipantKey", "action_code"}:
            continue
        if col in {"total_score", "score_norm", "confidence_score", "priority"}:
            agg_spec[col] = "max"
        else:
            agg_spec[col] = "first"

    deduped = (
        df.groupby(key_cols, as_index=False)
        .agg(agg_spec)
        .copy()
    )

    return deduped


def _add_rule_rank_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out = out.sort_values(
        by=["ParticipantKey", "total_score", "confidence_score", "priority", "action_code"],
        ascending=[True, False, False, False, True],
        kind="stable",
    ).reset_index(drop=True)

    out["rule_rank"] = out.groupby("ParticipantKey").cumcount() + 1
    out["candidate_count_for_participant"] = out.groupby("ParticipantKey")["action_code"].transform("count")

    if "bundle_family" in out.columns:
        out["bundle_rank_within_participant"] = (
            out.groupby(["ParticipantKey", "bundle_family"]).cumcount() + 1
        )
    else:
        out["bundle_rank_within_participant"] = 1

    return out


def _apply_label_scheme(
    df: pd.DataFrame,
    label_scheme: str,
    top_k_positive: int,
    min_score_norm: float,
) -> pd.DataFrame:
    out = df.copy()
    scheme = str(label_scheme).strip().lower()

    top_k_mask = out["rule_rank"] <= int(top_k_positive)

    if scheme == "rule_top_k":
        out["label"] = top_k_mask.astype(int)

    elif scheme == "rule_top_k_with_min_score":
        if "score_norm" not in out.columns:
            raise ValueError("score_norm is required for rule_top_k_with_min_score")
        out["label"] = (
            top_k_mask & (out["score_norm"].fillna(0.0) >= float(min_score_norm))
        ).astype(int)

    elif scheme == "rule_top_k_plus_bundle_retention":
        bundle_keep_mask = out["bundle_rank_within_participant"] == 1
        out["label"] = (top_k_mask | bundle_keep_mask).astype(int)

    else:
        raise ValueError(
            f"Unsupported label_scheme: {label_scheme}. "
            f"Supported: rule_top_k, rule_top_k_with_min_score, rule_top_k_plus_bundle_retention"
        )

    out["positive_count_for_participant"] = out.groupby("ParticipantKey")["label"].transform("sum")
    return out
