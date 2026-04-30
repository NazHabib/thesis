import pandas as pd

from apps.common.clinical_safety.temporal import build_temporal_features
from .contracts import validate_cfm_columns
from .feature_tiers import build_feature_tiers
from .safety_context import build_safety_context
from .candidates import explode_candidates, validate_participants_candidates_df
from .gating import apply_cold_start_gating


SAFETY_CONTEXT_COLUMNS = [
    "ParticipantKey",
    "Cycle",
    "bp_core",
    "a1c_core",
    "bmi_core",
    "diet_core",
    "feature_tier",
    "data_completeness_score",
    "triage_mode",
    "triage_reason",
    "bp_quality",
    "diet_day1_status",
    "confidence_ceiling_global",
    "ceiling_DietPattern",
    "ceiling_Movement",
    "ceiling_Weight",
    "ceiling_Smoking",
    "ceiling_Alcohol",
    "ceiling_Lipids",
    "ceiling_Monitoring",
    "ceiling_IntegratedCardiometabolic",
    "followup_urgency",
    "followup_horizon_days",
    "monitoring_need_level",
    "temporal_risk_tier",
    "persistence_risk_proxy",
    "repeat_measurement_available",
    "temporal_data_quality",
    "longitudinal_ready_flag",
    "warnings",
    "notes_pt",
    "cold_start_version",
]
GATED_CANDIDATE_COLUMNS = [
    "ParticipantKey",
    "action_code",
    "action_type",
    "severity",
    "priority",
    "tags",
    "evidence",
    "source_rule_ids",
    "bundle_family",
    "coldstart_gate",
    "coldstart_drop_reason",
    "generated_by_layer",
    "coldstart_confidence_ceiling",
    "coldstart_note_codes",
    "cold_start_flag",
]


def run_cold_start_layer(cfm_df: pd.DataFrame, participants_df: pd.DataFrame):
    validate_cfm_columns(cfm_df)
    validate_participants_candidates_df(participants_df)

    temporal_df = build_temporal_features(cfm_df)
    tiered_df = build_feature_tiers(temporal_df)
    safety_context_df = build_safety_context(tiered_df)
    candidates_df = explode_candidates(participants_df)

    if candidates_df.empty:
        gated_candidates_df = pd.DataFrame(columns=GATED_CANDIDATE_COLUMNS)
    else:
        missing_context = sorted(
            set(candidates_df["ParticipantKey"].astype(str))
            - set(safety_context_df["ParticipantKey"].astype(str))
        )
        if missing_context:
            preview = missing_context[:10]
            raise ValueError(
                "Missing safety context for candidate participants: "
                f"{preview}"
                + (" ..." if len(missing_context) > 10 else "")
            )

        gated_candidates_df = apply_cold_start_gating(candidates_df, safety_context_df)
        gated_candidates_df = gated_candidates_df[GATED_CANDIDATE_COLUMNS].copy()

    safety_context_df = safety_context_df[SAFETY_CONTEXT_COLUMNS].copy()
    
    n_participants = int(safety_context_df["ParticipantKey"].nunique())

    manifest = {
        "cold_start_version": safety_context_df["cold_start_version"].iloc[0]
        if not safety_context_df.empty and "cold_start_version" in safety_context_df.columns
        else "unknown",
        "participants": n_participants,
        "input_cfm_rows": int(len(cfm_df)),
        "input_participant_candidate_rows": int(len(participants_df)),
        "output_safety_context_rows": int(len(safety_context_df)),
        "output_gated_candidates": int(len(gated_candidates_df)),
        "triage_count": int(safety_context_df["triage_mode"].fillna(False).sum()),
        "feature_tier_counts": {
            "minimal": int((safety_context_df["feature_tier"] == "minimal").sum()),
            "core": int((safety_context_df["feature_tier"] == "core").sum()),
            "full": int((safety_context_df["feature_tier"] == "full").sum()),
        },
        "confidence_ceiling_global_counts": {
            str(k): int(v)
            for k, v in safety_context_df["confidence_ceiling_global"]
            .fillna("unknown")
            .value_counts(dropna=False)
            .to_dict()
            .items()
        },
        "followup_urgency_counts": {
            str(k): int(v)
            for k, v in safety_context_df["followup_urgency"]
            .fillna("unknown")
            .value_counts(dropna=False)
            .to_dict()
            .items()
        },
        "temporal_risk_tier_counts": {
            str(k): int(v)
            for k, v in safety_context_df["temporal_risk_tier"]
            .fillna("unknown")
            .value_counts(dropna=False)
            .to_dict()
            .items()
        },
        "gated_bundle_family_counts": {
            str(k): int(v)
            for k, v in gated_candidates_df["bundle_family"]
            .fillna("unknown")
            .value_counts(dropna=False)
            .to_dict()
            .items()
        } if not gated_candidates_df.empty else {},
    }

    return {
        "safety_context": safety_context_df,
        "gated_candidates": gated_candidates_df,
        "manifest": manifest,
    }
