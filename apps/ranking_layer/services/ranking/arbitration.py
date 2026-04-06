import pandas as pd

from .config import MAX_RECOMMENDATIONS_PER_PARTICIPANT

MAX_PER_FAMILY = 2


def apply_conflict_policy(group: pd.DataFrame) -> pd.DataFrame:
    if "conflict_policy" not in group.columns or group["conflict_policy"].isna().all():
        return group

    policy = group["conflict_policy"].dropna().iloc[0]

    if policy == "keep_strongest":
        return group.sort_values("total_score", ascending=False).head(1)

    if policy == "mutually_exclusive":
        return group.sort_values("total_score", ascending=False).head(1)

    if policy == "suppress_recommend_if_escalate":
        if (group["action_type"] == "escalate").any():
            return group[group["action_type"] == "escalate"]

    return group


def apply_family_cap(group: pd.DataFrame) -> pd.DataFrame:
    if "bundle_family" not in group.columns:
        return group.head(MAX_RECOMMENDATIONS_PER_PARTICIPANT)

    capped = (
        group.sort_values(
            ["total_score", "base_severity_score", "priority_num", "action_code"],
            ascending=[False, False, False, True],
        )
        .groupby("bundle_family", sort=False, group_keys=False)
        .head(MAX_PER_FAMILY)
    )

    capped = capped.sort_values(
        ["total_score", "base_severity_score", "priority_num", "action_code"],
        ascending=[False, False, False, True],
    )

    return capped.head(MAX_RECOMMENDATIONS_PER_PARTICIPANT)


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

    return out
