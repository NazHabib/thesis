import pandas as pd

from .config import BUNDLING_VERSION
from .contracts import validate_safety_context_columns, validate_ranked_columns
from .bundler import build_bundles


FINAL_BUNDLE_COLUMNS = [
    "ParticipantKey",
    "bundle_title",
    "bundle_family",
    "category",
    "action_codes",
    "action_type",
    "priority",
    "tags",
    "source_rule_ids",
    "evidence",
    "coldstart_note_codes",
    "bundle_score",
    "confidence_score",
    "confidence_label",
    "rationale",
    "explanation",
    "triage_mode",
    "triage_reason",
    "followup_urgency",
    "followup_horizon_days",
    "monitoring_need_level",
    "temporal_risk_tier",
    "persistence_risk_proxy",
    "temporal_data_quality",
    "longitudinal_ready_flag",
    "n_actions_in_bundle",
    "dominant_action_code",
    "bundle_rank",
    "followup_window_label",
    "followup_window_order",
    "followup_timing_text",
    "fusion_strategy",
    "fusion_version",
    "fusion_status",
    "rule_score_norm_participant",
    "fusion_rule_component",
    "fusion_ml_component",
]


def run_bundling_layer(
    safety_context_df: pd.DataFrame,
    ranked_df: pd.DataFrame,
) -> dict:
    validate_safety_context_columns(safety_context_df)
    validate_ranked_columns(ranked_df)

    merged_df = ranked_df.merge(
        safety_context_df,
        on="ParticipantKey",
        how="left",
        validate="m:1",
    )

    if "final_score" in merged_df.columns:
        merged_df["rule_total_score"] = merged_df["total_score"]
        merged_df["effective_score"] = merged_df["final_score"]
    else:
        merged_df["rule_total_score"] = merged_df["total_score"]
        merged_df["effective_score"] = merged_df["total_score"]

    bundles_df = build_bundles(merged_df)

    if "action_codes" in bundles_df.columns:
        bundles_df["n_actions_in_bundle"] = bundles_df["action_codes"].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )
        bundles_df["dominant_action_code"] = bundles_df["action_codes"].apply(
            lambda x: x[0] if isinstance(x, list) and x else None
        )
    else:
        bundles_df["n_actions_in_bundle"] = 0
        bundles_df["dominant_action_code"] = None

    sort_cols = [c for c in [
        "ParticipantKey",
        "bundle_score",
        "confidence_score",
        "followup_window_order",
        "followup_horizon_days",
        "priority",
        "bundle_title",
    ] if c in bundles_df.columns]
    ascending = [True, False, False, True, True, False, True][:len(sort_cols)]
    bundles_df = bundles_df.sort_values(by=sort_cols, ascending=ascending, kind="stable").reset_index(drop=True)

    bundles_df["bundle_rank"] = bundles_df.groupby("ParticipantKey").cumcount() + 1

    bundles_df = bundles_df[FINAL_BUNDLE_COLUMNS].copy()

    n_participants = int(safety_context_df["ParticipantKey"].nunique())

    manifest = {
        "bundling_version": BUNDLING_VERSION,
        "participants": n_participants,
        "input_ranked_candidates": int(len(ranked_df)),
        "output_final_recommendations": int(len(bundles_df)),
        "triage_count": int(safety_context_df["triage_mode"].fillna(False).sum()),
        "avg_bundles_per_participant": round(len(bundles_df) / max(1, n_participants), 4),
        "avg_actions_per_bundle": round(float(bundles_df["n_actions_in_bundle"].mean()), 4) if not bundles_df.empty else 0.0,
        "bundle_family_counts": {
            str(k): int(v)
            for k, v in bundles_df["bundle_family"].value_counts(dropna=False).to_dict().items()
        },
        "confidence_counts": {
            "high": int((bundles_df["confidence_label"] == "high").sum()),
            "medium": int((bundles_df["confidence_label"] == "medium").sum()),
            "low": int((bundles_df["confidence_label"] == "low").sum()),
        },
        "followup_window_counts": {
            str(k): int(v)
            for k, v in bundles_df["followup_window_label"].fillna("unknown").value_counts(dropna=False).to_dict().items()
        } if "followup_window_label" in bundles_df.columns else {},
        "followup_urgency_counts": {
            str(k): int(v)
            for k, v in bundles_df["followup_urgency"].fillna("unknown").value_counts(dropna=False).to_dict().items()
        } if "followup_urgency" in bundles_df.columns else {},
        "temporal_risk_tier_counts": {
            str(k): int(v)
            for k, v in bundles_df["temporal_risk_tier"].fillna("unknown").value_counts(dropna=False).to_dict().items()
        } if "temporal_risk_tier" in bundles_df.columns else {},
        "score_summary": {
            "min_bundle_score": float(bundles_df["bundle_score"].min()) if not bundles_df.empty else 0.0,
            "max_bundle_score": float(bundles_df["bundle_score"].max()) if not bundles_df.empty else 0.0,
            "mean_bundle_score": round(float(bundles_df["bundle_score"].mean()), 4) if not bundles_df.empty else 0.0,
        },
    }

    return {
        "final_recommendations": bundles_df,
        "manifest": manifest,
    }
