import pandas as pd

from .constants import COLLAB_DEFAULTS, COLLAB_FEATURE_COLUMNS


def build_collaborative_features(
    ranked_candidates_df: pd.DataFrame,
    neighbors_df: pd.DataFrame,
    reference_ranked_df: pd.DataFrame,
    min_neighbors: int = 5,
) -> pd.DataFrame:
    candidates = ranked_candidates_df[
        ["ParticipantKey", "action_code", "bundle_family"]
    ].drop_duplicates().copy()

    if candidates.empty:
        return pd.DataFrame(columns=COLLAB_FEATURE_COLUMNS)

    if neighbors_df.empty or reference_ranked_df.empty:
        return _with_defaults(candidates)

    ref = reference_ranked_df.copy()

    if "rank_pre_ml" not in ref.columns:
        sort_cols = [
            c for c in [
                "ParticipantKey",
                "score_norm",
                "confidence_score",
                "priority",
                "action_code",
            ]
            if c in ref.columns
        ]
        ascending = [True, False, False, False, True][:len(sort_cols)]

        ref = ref.sort_values(
            sort_cols,
            ascending=ascending,
            kind="stable",
        ).copy()

        ref["rank_pre_ml"] = ref.groupby("ParticipantKey").cumcount() + 1

    joined = neighbors_df.merge(
        ref[
            [
                "ParticipantKey",
                "action_code",
                "bundle_family",
                "score_norm",
                "rank_pre_ml",
            ]
        ],
        left_on="neighbor_participant_key",
        right_on="ParticipantKey",
        how="left",
        suffixes=("_target", "_neighbor"),
    )

    if "ParticipantKey_target" in joined.columns:
        joined = joined.rename(columns={"ParticipantKey_target": "ParticipantKey"})
    elif "ParticipantKey_x" in joined.columns:
        joined = joined.rename(columns={"ParticipantKey_x": "ParticipantKey"})

    if "ParticipantKey_neighbor" in joined.columns:
        joined = joined.rename(columns={"ParticipantKey_neighbor": "neighbor_key"})
    elif "ParticipantKey_y" in joined.columns:
        joined = joined.rename(columns={"ParticipantKey_y": "neighbor_key"})

    joined["similarity"] = pd.to_numeric(joined["similarity"], errors="coerce").fillna(0.0)
    joined["score_norm"] = pd.to_numeric(joined["score_norm"], errors="coerce")
    joined["rank_pre_ml"] = pd.to_numeric(joined["rank_pre_ml"], errors="coerce")

    neighbor_counts = (
        neighbors_df.groupby("ParticipantKey")["neighbor_participant_key"]
        .nunique()
        .reset_index(name="collab_neighbor_count")
    )

    exact = _build_exact_action_features(joined)
    family = _build_family_features(joined)

    out = candidates.copy()
    out = out.merge(neighbor_counts, on="ParticipantKey", how="left")
    out = out.merge(exact, on=["ParticipantKey", "action_code"], how="left")
    out = out.merge(family, on=["ParticipantKey", "bundle_family"], how="left")

    out["collab_neighbor_count"] = out["collab_neighbor_count"].fillna(0).astype(int)
    out["collab_support_count"] = out["collab_support_count"].fillna(0).astype(int)
    out["collab_family_support_count"] = out["collab_family_support_count"].fillna(0).astype(int)

    out["collab_support_rate"] = (
        out["collab_support_count"] / out["collab_neighbor_count"].replace(0, pd.NA)
    ).fillna(0.0)

    out["collab_family_support_rate"] = (
        out["collab_family_support_count"] / out["collab_neighbor_count"].replace(0, pd.NA)
    ).fillna(0.0)

    out["collab_available_flag"] = (
        (out["collab_neighbor_count"] >= int(min_neighbors))
        & (
            (out["collab_support_count"] > 0)
            | (out["collab_family_support_count"] > 0)
        )
    ).astype(int)

    out["collab_safety_interpretation"] = out.apply(
        _build_safety_interpretation,
        axis=1,
    )

    out = out.drop(columns=["collab_family_support_count"], errors="ignore")

    return _with_defaults(out)


def _build_exact_action_features(joined: pd.DataFrame) -> pd.DataFrame:
    if joined.empty:
        return pd.DataFrame(
            columns=[
                "ParticipantKey",
                "action_code",
                "collab_support_count",
                "collab_mean_rule_score_norm",
                "collab_mean_rank_pre_ml",
                "collab_weighted_support_rate",
                "collab_weighted_mean_rule_score_norm",
                "collab_weighted_mean_rank_pre_ml",
                "collab_mean_supporting_similarity",
            ]
        )

    support = joined.dropna(subset=["action_code"]).copy()

    if support.empty:
        return pd.DataFrame(
            columns=[
                "ParticipantKey",
                "action_code",
                "collab_support_count",
                "collab_mean_rule_score_norm",
                "collab_mean_rank_pre_ml",
                "collab_weighted_support_rate",
                "collab_weighted_mean_rule_score_norm",
                "collab_weighted_mean_rank_pre_ml",
                "collab_mean_supporting_similarity",
            ]
        )

    grouped = (
        support.groupby(["ParticipantKey", "action_code"], dropna=False)
        .apply(_aggregate_weighted_action)
        .reset_index()
    )

    return grouped


def _aggregate_weighted_action(group: pd.DataFrame) -> pd.Series:
    weights = pd.to_numeric(group["similarity"], errors="coerce").fillna(0.0)
    score = pd.to_numeric(group["score_norm"], errors="coerce")
    rank = pd.to_numeric(group["rank_pre_ml"], errors="coerce")

    weight_sum = float(weights.sum())

    if weight_sum > 0:
        weighted_score = float((score.fillna(0.0) * weights).sum() / weight_sum)
        weighted_rank = float((rank.fillna(rank.median() if rank.notna().any() else 0.0) * weights).sum() / weight_sum)
    else:
        weighted_score = 0.0
        weighted_rank = 0.0

    return pd.Series(
        {
            "collab_support_count": int(group["neighbor_participant_key"].nunique()),
            "collab_mean_rule_score_norm": float(score.mean()) if score.notna().any() else 0.0,
            "collab_mean_rank_pre_ml": float(rank.mean()) if rank.notna().any() else 0.0,
            "collab_weighted_support_rate": weight_sum,
            "collab_weighted_mean_rule_score_norm": weighted_score,
            "collab_weighted_mean_rank_pre_ml": weighted_rank,
            "collab_mean_supporting_similarity": float(weights.mean()) if len(weights) else 0.0,
        }
    )


def _build_family_features(joined: pd.DataFrame) -> pd.DataFrame:
    if joined.empty or "bundle_family" not in joined.columns:
        return pd.DataFrame(
            columns=[
                "ParticipantKey",
                "bundle_family",
                "collab_family_support_count",
            ]
        )

    family = (
        joined.dropna(subset=["bundle_family"])
        .groupby(["ParticipantKey", "bundle_family"], dropna=False)
        .agg(
            collab_family_support_count=("neighbor_participant_key", "nunique"),
        )
        .reset_index()
    )

    return family


def _build_safety_interpretation(row: pd.Series) -> str:
    if int(row.get("collab_available_flag", 0) or 0) != 1:
        return (
            "Collaborative support was unavailable or insufficient; this signal did not "
            "introduce or remove any recommendation."
        )

    support_count = int(row.get("collab_support_count", 0) or 0)
    neighbor_count = int(row.get("collab_neighbor_count", 0) or 0)
    mean_rank = float(row.get("collab_mean_rank_pre_ml", 0.0) or 0.0)

    if support_count > 0:
        return (
            f"Among similar patients, this candidate appeared in {support_count} of "
            f"{neighbor_count} neighbour recommendation sets, with an average pre-ML "
            f"rank of {mean_rank:.2f}. Collaborative filtering was used only as a "
            "support signal and did not introduce new actions."
        )

    return (
        f"Similar patients showed support for the same recommendation family among "
        f"{neighbor_count} neighbours. Collaborative filtering was used only as a "
        "support signal and did not introduce new actions."
    )


def _with_defaults(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col, default in COLLAB_DEFAULTS.items():
        if col not in out.columns:
            out[col] = default
        else:
            out[col] = out[col].fillna(default)

    for col in ["ParticipantKey", "action_code", "bundle_family"]:
        if col not in out.columns:
            out[col] = None

    # Convert weighted support from raw similarity sum into a true rate
    if {
        "collab_weighted_support_rate",
        "collab_neighbor_count",
    }.issubset(out.columns):
        denom = out["collab_neighbor_count"].replace(0, pd.NA)
        out["collab_weighted_support_rate"] = (
            out["collab_weighted_support_rate"] / denom
        ).fillna(0.0)

    numeric_cols = [
        c for c in COLLAB_DEFAULTS
        if c != "collab_safety_interpretation"
    ]

    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(COLLAB_DEFAULTS[col])

    return out[COLLAB_FEATURE_COLUMNS].copy()
