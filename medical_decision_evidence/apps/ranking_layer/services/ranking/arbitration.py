import pandas as pd

from .config import MAX_RECOMMENDATIONS_PER_PARTICIPANT

MAX_PER_FAMILY = 2


def _sort_candidate_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "priority_num" not in out.columns:
        out["priority_num"] = pd.to_numeric(out["priority"], errors="coerce").fillna(0)

    return out.sort_values(
        ["total_score", "base_severity_score", "priority_num", "action_code"],
        ascending=[False, False, False, True],
        kind="stable",
    )


def apply_conflict_policy(group: pd.DataFrame) -> pd.DataFrame:
    if "conflict_policy" not in group.columns or group["conflict_policy"].isna().all():
        return group

    policies = group["conflict_policy"].dropna().astype(str).unique().tolist()
    if len(policies) > 1:
        raise ValueError(f"Mixed conflict_policy values within one conflict group: {policies}")

    policy = policies[0]
    action_type = group["action_type"].fillna("").astype(str)

    if policy in {"keep_strongest", "mutually_exclusive"}:
        return _sort_candidate_frame(group).head(1)

    if policy == "suppress_recommend_if_escalate":
        if action_type.eq("escalate").any():
            return _sort_candidate_frame(group[action_type.eq("escalate")])

    return group


def apply_family_cap(group: pd.DataFrame) -> pd.DataFrame:
    sorted_group = _sort_candidate_frame(group)

    safety_mask = sorted_group["action_type"].astype("string").isin({"escalate", "forbid"})
    safety_items = sorted_group[safety_mask]
    normal_items = sorted_group[~safety_mask]

    if "bundle_family" not in normal_items.columns:
        capped_normal = _sort_candidate_frame(normal_items).head(MAX_RECOMMENDATIONS_PER_PARTICIPANT)
    else:
        capped_normal = (
            _sort_candidate_frame(normal_items)
            .groupby("bundle_family", sort=False, group_keys=False)
            .head(MAX_PER_FAMILY)
        )

    combined = pd.concat([safety_items, capped_normal], ignore_index=True, sort=False)
    combined = _sort_candidate_frame(combined)

    if safety_items.empty:
        return combined.head(MAX_RECOMMENDATIONS_PER_PARTICIPANT)

    normal_slots = max(0, MAX_RECOMMENDATIONS_PER_PARTICIPANT - len(safety_items))
    capped_normal = _sort_candidate_frame(capped_normal).head(normal_slots)

    combined = pd.concat([safety_items, capped_normal], ignore_index=True, sort=False)
    return _sort_candidate_frame(combined)


def arbitrate_candidates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "conflict_group" in out.columns:
        groups_with_conflict = out["conflict_group"].notna()
        conflicted = out[groups_with_conflict]
        non_conflicted = out[~groups_with_conflict]

        if not conflicted.empty:
            conflicted_parts = []
            for (_, _), grp in conflicted.groupby(["ParticipantKey", "conflict_group"], sort=False):
                conflicted_parts.append(apply_conflict_policy(grp))

            conflicted = (
                pd.concat(conflicted_parts, ignore_index=True, sort=False)
                if conflicted_parts
                else conflicted.iloc[0:0].copy()
            )

        out = pd.concat([conflicted, non_conflicted], ignore_index=True, sort=False)

    out = out.sort_values(
        ["ParticipantKey", "total_score", "base_severity_score", "priority_num", "action_code"],
        ascending=[True, False, False, False, True],
        kind="stable",
    )

    ranked_parts = []
    for participant_key, grp in out.groupby("ParticipantKey", sort=False):
        capped = apply_family_cap(grp.copy())
        capped["ParticipantKey"] = participant_key
        ranked_parts.append(capped)

    if ranked_parts:
        out = pd.concat(ranked_parts, ignore_index=True, sort=False)
    else:
        out = out.iloc[0:0].copy()

    if not out.empty:
        out = out.sort_values(
            ["ParticipantKey", "total_score", "base_severity_score", "priority_num", "action_code"],
            ascending=[True, False, False, False, True],
            kind="stable",
        ).reset_index(drop=True)

    return out
