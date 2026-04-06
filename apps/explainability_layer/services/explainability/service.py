import pandas as pd

from .config import EXPLAINABILITY_VERSION
from .contracts import validate_final_bundle_columns, validate_safety_context_columns
from .explainers import explain_bundle


FINAL_EXPLAINED_COLUMNS = [
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
    "patient_facing_explanation",
    "clinician_facing_explanation",
    "patient_confidence_explanation",
    "clinician_confidence_explanation",
    "safety_context_note",
    "ml_rerank_note",
    "clinician_ml_xai_explanation",
    "patient_ml_xai_explanation",
    "ml_xai_payload",
    "traceability_payload",
]


def run_explainability_layer(
    safety_context_df: pd.DataFrame,
    final_recommendations_df: pd.DataFrame,
    reranked_df: pd.DataFrame | None = None,
    participant_features_df: pd.DataFrame | None = None,
) -> dict:
    validate_safety_context_columns(safety_context_df)
    validate_final_bundle_columns(final_recommendations_df)

    merged_df = final_recommendations_df.merge(
        safety_context_df[
            ["ParticipantKey", "feature_tier", "triage_mode", "triage_reason", "confidence_ceiling_global"]
        ].drop_duplicates(subset=["ParticipantKey"]),
        on="ParticipantKey",
        how="left",
        suffixes=("", "_ctx"),
        validate="m:1",
    )

    explained_records = []
    for _, row in merged_df.iterrows():
        base = row.to_dict()
        explanation_payload = explain_bundle(
            base,
            reranked_df=reranked_df,
            safety_context_df=safety_context_df,
            participant_features_df=participant_features_df,
        )
        base.update(explanation_payload)
        explained_records.append(base)

    explained_df = pd.DataFrame(explained_records)

    if "triage_mode_ctx" in explained_df.columns:
        explained_df = explained_df.drop(columns=["triage_mode_ctx"])
    if "triage_reason_ctx" in explained_df.columns:
        explained_df = explained_df.drop(columns=["triage_reason_ctx"])

    explained_df = explained_df[FINAL_EXPLAINED_COLUMNS].copy()

    n_participants = int(safety_context_df["ParticipantKey"].nunique())

    manifest = {
        "explainability_version": EXPLAINABILITY_VERSION,
        "participants": n_participants,
        "input_final_recommendations": int(len(final_recommendations_df)),
        "output_explained_recommendations": int(len(explained_df)),
        "triage_count": int(safety_context_df["triage_mode"].fillna(False).sum()),
        "confidence_counts": {
            "high": int((explained_df["confidence_label"] == "high").sum()),
            "medium": int((explained_df["confidence_label"] == "medium").sum()),
            "low": int((explained_df["confidence_label"] == "low").sum()),
        },
        "bundle_family_counts": {
            str(k): int(v)
            for k, v in explained_df["bundle_family"].value_counts(dropna=False).to_dict().items()
        },
        "xai_status_counts": {
            str(k): int(v)
            for k, v in explained_df["ml_xai_payload"].apply(lambda x: x.get("xai_status", "unknown")).value_counts(dropna=False).to_dict().items()
        },
    }

    return {
        "explained_recommendations": explained_df,
        "manifest": manifest,
    }
