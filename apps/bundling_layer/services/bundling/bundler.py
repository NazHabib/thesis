import pandas as pd

from .config import MAX_FINAL_BUNDLES_PER_PARTICIPANT
from .family_map import infer_bundle_family
from .helpers import (
    build_bundle_category,
    build_bundle_title,
    build_explanation,
    build_rationale,
    build_bundle_followup_window_label,
    build_bundle_followup_timing_text,
    conservative_confidence,
    merge_evidence_lists,
    unique_flatten,
)
from apps.common.clinical_safety.scheduling import derive_followup_window_order

def aggregate_bundle(
    participant_key: str,
    bundle_family: str,
    group: pd.DataFrame,
) -> pd.Series:
    triage_mode = bool(group["triage_mode"].fillna(False).iloc[0])

    action_codes = group["action_code"].dropna().astype(str).unique().tolist()
    tags = unique_flatten(group["tags"].tolist())
    source_rule_ids = unique_flatten(group["source_rule_ids"].tolist())
    coldstart_note_codes = unique_flatten(group["coldstart_note_codes"].tolist())
    evidence = merge_evidence_lists(group["evidence"])

    action_types = set(group["action_type"].dropna().astype(str).str.lower())

    if "escalate" in action_types:
        bundle_action_type = "escalate"
    elif "forbid" in action_types:
        bundle_action_type = "forbid"
    elif "recommend" in action_types:
        bundle_action_type = "recommend"
    elif "inform" in action_types:
        bundle_action_type = "inform"
    else:
        bundle_action_type = "recommend"

    final_family = (
        str(bundle_family).strip()
        if bundle_family is not None and pd.notna(bundle_family) and str(bundle_family).strip()
        else infer_bundle_family(action_codes, tags)
    )

    bundle_priority = int(pd.to_numeric(group["priority"], errors="coerce").fillna(0).max())
    score_col = "effective_score" if "effective_score" in group.columns else "total_score"
    bundle_score = float(pd.to_numeric(group[score_col], errors="coerce").fillna(0).max())
    fusion_strategy = group["fusion_strategy"].dropna().astype(str).iloc[0] if "fusion_strategy" in group.columns and not group["fusion_strategy"].dropna().empty else None
    fusion_version = group["fusion_version"].dropna().astype(str).iloc[0] if "fusion_version" in group.columns and not group["fusion_version"].dropna().empty else None
    fusion_status = group["fusion_status"].dropna().astype(str).iloc[0] if "fusion_status" in group.columns and not group["fusion_status"].dropna().empty else None
    rule_score_norm_participant = float(pd.to_numeric(group["rule_score_norm_participant"], errors="coerce").fillna(0).max()) if "rule_score_norm_participant" in group.columns else 0.0
    fusion_rule_component = float(pd.to_numeric(group["fusion_rule_component"], errors="coerce").fillna(0).max()) if "fusion_rule_component" in group.columns else 0.0
    fusion_ml_component = float(pd.to_numeric(group["fusion_ml_component"], errors="coerce").fillna(0).max()) if "fusion_ml_component" in group.columns else 0.0
    confidence_label = conservative_confidence(group["confidence_label"])
    confidence_score = float(
        pd.to_numeric(group["confidence_score"], errors="coerce").fillna(0).mean()
    )

    followup_urgency = group["followup_urgency"].iloc[0] if "followup_urgency" in group.columns else None
    followup_horizon_days = (
        pd.to_numeric(group["followup_horizon_days"], errors="coerce").dropna().min()
        if "followup_horizon_days" in group.columns
        else None
    )
    monitoring_need_level = group["monitoring_need_level"].iloc[0] if "monitoring_need_level" in group.columns else None
    temporal_risk_tier = group["temporal_risk_tier"].iloc[0] if "temporal_risk_tier" in group.columns else None
    persistence_risk_proxy = (
        pd.to_numeric(group["persistence_risk_proxy"], errors="coerce").dropna().max()
        if "persistence_risk_proxy" in group.columns
        else None
    )
    temporal_data_quality = group["temporal_data_quality"].iloc[0] if "temporal_data_quality" in group.columns else None
    longitudinal_ready_flag = (
        int(pd.to_numeric(group["longitudinal_ready_flag"], errors="coerce").fillna(0).max())
        if "longitudinal_ready_flag" in group.columns
        else 0
    )
    
    followup_window_label = build_bundle_followup_window_label(
        followup_horizon_days=followup_horizon_days,
        followup_urgency=followup_urgency,
    )
    followup_window_order = derive_followup_window_order(followup_window_label)
    followup_timing_text = build_bundle_followup_timing_text(
        followup_window_label=followup_window_label,
        monitoring_need_level=monitoring_need_level,
        bundle_family=final_family,
    )
    return pd.Series(
        {
            "ParticipantKey": participant_key,
            "bundle_family": final_family,
            "bundle_title": build_bundle_title(final_family, bundle_action_type, action_codes, tags),
            "category": build_bundle_category(final_family),
            "action_codes": action_codes,
            "action_type": bundle_action_type,
            "priority": bundle_priority,
            "tags": tags,
            "source_rule_ids": source_rule_ids,
            "evidence": evidence,
            "coldstart_note_codes": coldstart_note_codes,
            "bundle_score": bundle_score,
            "confidence_score": round(confidence_score, 4),
            "confidence_label": confidence_label,
            "rationale": build_rationale(final_family, tags, triage_mode),
            "explanation": build_explanation(final_family, action_codes, confidence_label, tags),
            "followup_urgency": followup_urgency,
            "followup_horizon_days": followup_horizon_days,
            "monitoring_need_level": monitoring_need_level,
            "temporal_risk_tier": temporal_risk_tier,
            "persistence_risk_proxy": persistence_risk_proxy,
            "temporal_data_quality": temporal_data_quality,
            "longitudinal_ready_flag": longitudinal_ready_flag,
            "followup_window_label": followup_window_label,
            "followup_window_order": followup_window_order,
            "followup_timing_text": followup_timing_text,
            "fusion_strategy": fusion_strategy,
            "fusion_version": fusion_version,
            "fusion_status": fusion_status,
            "rule_score_norm_participant": rule_score_norm_participant,
            "fusion_rule_component": fusion_rule_component,
            "fusion_ml_component": fusion_ml_component,
        }
    )


def apply_triage_policy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    triage_mask = out["triage_mode"].fillna(False)
    triage_df = out[triage_mask].copy()
    non_triage_df = out[~triage_mask].copy()

    if triage_df.empty:
        return out

    triage_parts = []
    for participant_key, grp in triage_df.groupby("ParticipantKey", sort=False):
        escalates = grp[grp["action_type"] == "escalate"].copy()

        if not escalates.empty:
            keep = escalates.sort_values(
                ["bundle_score", "confidence_score", "followup_window_order", "followup_horizon_days", "priority", "bundle_title"],
                ascending=[False, False, True, True, True, True],
                kind="stable",
            ).head(1)
        else:
            keep = grp.sort_values(
                ["bundle_score", "confidence_score", "followup_window_order", "followup_horizon_days", "priority", "bundle_title"],
                ascending=[False, False, True, True, True, True],
                kind="stable",
            ).head(1)

        triage_parts.append(keep)

    triage_kept = (
        pd.concat(triage_parts, ignore_index=True, sort=False)
        if triage_parts
        else triage_df.iloc[0:0].copy()
    )

    return pd.concat([non_triage_df, triage_kept], ignore_index=True, sort=False)


def select_top_bundles(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(
        ["ParticipantKey", "bundle_score", "confidence_score", "followup_window_order", "followup_horizon_days", "priority", "bundle_title"],
        ascending=[True, False, False, True, True, True, True],
        kind="stable",
    ).copy()

    return out.groupby("ParticipantKey", sort=False).head(MAX_FINAL_BUNDLES_PER_PARTICIPANT).copy()


def build_bundles(merged_df: pd.DataFrame) -> pd.DataFrame:
    # Remove internal/system-only audit messages from patient-facing bundles
    merged_df = merged_df[
        ~merged_df["action_code"].astype(str).eq("INFORM_SYSTEM_SCOPE")
    ].copy()

    bundle_parts = []

    for (participant_key, bundle_family), grp in merged_df.groupby(
        ["ParticipantKey", "bundle_family"],
        sort=False,
        dropna=False,
    ):
        family_value = bundle_family if pd.notna(bundle_family) else "IntegratedCardiometabolic"
        bundle_parts.append(aggregate_bundle(participant_key, family_value, grp.copy()))

    grouped = (
        pd.DataFrame(bundle_parts)
        if bundle_parts
        else pd.DataFrame(
            columns=[
                "ParticipantKey",
                "bundle_family",
                "bundle_title",
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
                "followup_urgency",
                "followup_horizon_days",
                "followup_window_label",
                "followup_window_order",
                "followup_timing_text",
                "monitoring_need_level",
                "temporal_risk_tier",
                "persistence_risk_proxy",
                "temporal_data_quality",
                "longitudinal_ready_flag",
            ]
        )
    )

    participant_flags = merged_df[
        ["ParticipantKey", "triage_mode", "triage_reason"]
    ].drop_duplicates(subset=["ParticipantKey"])

    grouped = grouped.merge(
        participant_flags,
        on="ParticipantKey",
        how="left",
        validate="m:1",
    )

    grouped = apply_triage_policy(grouped)
    grouped = select_top_bundles(grouped)

    return grouped
