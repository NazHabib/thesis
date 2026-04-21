import pandas as pd

from .config import RANKING_VERSION
from .contracts import validate_safety_context_columns, validate_gated_candidate_columns
from .bundle_map import apply_bundle_family
from .scoring import score_candidates
from .confidence import apply_confidence
from .arbitration import arbitrate_candidates


FINAL_RANKED_COLUMNS = [
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
    "positive_score_components",
    "negative_score_components",
    "ranking_rationale_short",
    "total_score",
    "score_norm",
    "confidence_score",
    "confidence_label",
]


def run_ranking_layer(
    safety_context_df: pd.DataFrame,
    gated_candidates_df: pd.DataFrame,
) -> dict:
    validate_safety_context_columns(safety_context_df)
    validate_gated_candidate_columns(gated_candidates_df)

    merged_df = gated_candidates_df.merge(
        safety_context_df,
        on="ParticipantKey",
        how="left",
        validate="m:1",
    )
    
    if not merged_df.empty:
        missing_context_mask = merged_df["feature_tier"].isna()
        if missing_context_mask.any():
            missing_keys = (
                merged_df.loc[missing_context_mask, "ParticipantKey"]
                .astype(str)
                .drop_duplicates()
                .tolist()
            )
            preview = missing_keys[:10]
            raise ValueError(
                "Ranking layer found candidates without matching safety_context rows: "
                f"{preview}"
                + (" ..." if len(missing_keys) > 10 else "")
            )

    merged_df = apply_bundle_family(merged_df)

    scored_df = score_candidates(merged_df)
    confident_df = apply_confidence(scored_df)
    ranked_df = arbitrate_candidates(confident_df)

    missing_final = [c for c in FINAL_RANKED_COLUMNS if c not in ranked_df.columns]
    if missing_final:
        raise ValueError(f"Ranking layer missing expected output columns: {missing_final}")

    ranked_df = ranked_df[FINAL_RANKED_COLUMNS].copy()

    n_participants = int(safety_context_df["ParticipantKey"].nunique())

    manifest = {
        "ranking_version": RANKING_VERSION,
        "participants": n_participants,
        "input_candidates": int(len(gated_candidates_df)),
        "output_ranked_candidates": int(len(ranked_df)),
        "triage_count": int(safety_context_df["triage_mode"].fillna(False).sum()),
        "feature_tier_counts": {
            "minimal": int((safety_context_df["feature_tier"] == "minimal").sum()),
            "core": int((safety_context_df["feature_tier"] == "core").sum()),
            "full": int((safety_context_df["feature_tier"] == "full").sum()),
        },
        "avg_ranked_per_participant": round(len(ranked_df) / max(1, n_participants), 4),
        "confidence_counts": {
            "high": int((ranked_df["confidence_label"] == "high").sum()),
            "medium": int((ranked_df["confidence_label"] == "medium").sum()),
            "low": int((ranked_df["confidence_label"] == "low").sum()),
        },
        "severity_counts": {
            str(k): int(v)
            for k, v in ranked_df["severity"].value_counts(dropna=False).to_dict().items()
        },
        "action_type_counts": {
            str(k): int(v)
            for k, v in ranked_df["action_type"].value_counts(dropna=False).to_dict().items()
        },
        "bundle_family_counts": {
            str(k): int(v)
            for k, v in ranked_df["bundle_family"].value_counts(dropna=False).to_dict().items()
        },
        "score_summary": {
            "min_total_score": float(ranked_df["total_score"].min()) if not ranked_df.empty else 0.0,
            "max_total_score": float(ranked_df["total_score"].max()) if not ranked_df.empty else 0.0,
            "mean_total_score": round(float(ranked_df["total_score"].mean()), 4) if not ranked_df.empty else 0.0,
        },
    }

    return {
        "ranked_recommendations": ranked_df,
        "manifest": manifest,
    }
