import pandas as pd

from .config import MAX_FINAL_BUNDLES_PER_PARTICIPANT
from .family_map import infer_bundle_family
from .helpers import (
    build_bundle_category,
    build_bundle_title,
    build_explanation,
    build_rationale,
    conservative_confidence,
    merge_evidence_lists,
    unique_flatten,
)


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

    if (group["action_type"] == "escalate").any():
        bundle_action_type = "escalate"
    else:
        mode_vals = group["action_type"].mode()
        bundle_action_type = mode_vals.iloc[0] if not mode_vals.empty else "recommend"

    final_family = infer_bundle_family(action_codes, tags)

    bundle_priority = int(pd.to_numeric(group["priority"], errors="coerce").fillna(999).min())
    bundle_score = float(pd.to_numeric(group["total_score"], errors="coerce").fillna(0).max())
    confidence_label = conservative_confidence(group["confidence_label"])
    confidence_score = float(
        pd.to_numeric(group["confidence_score"], errors="coerce").fillna(0).mean()
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
                ["bundle_score", "priority", "bundle_title"],
                ascending=[False, True, True],
            ).head(1)
        else:
            keep = grp.sort_values(
                ["bundle_score", "priority", "bundle_title"],
                ascending=[False, True, True],
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
        ["ParticipantKey", "bundle_score", "priority", "bundle_title"],
        ascending=[True, False, True, True],
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
