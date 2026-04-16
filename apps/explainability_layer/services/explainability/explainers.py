import pandas as pd

from .confidence import build_confidence_messages
from .templates import build_patient_why_text, build_clinician_why_text
from .traceability import build_traceability_payload
from .ml_xai import build_bundle_ml_xai_payload, build_ml_xai_texts


def explain_bundle(
    row,
    reranked_df=None,
    safety_context_df=None,
    participant_features_df=None,
) -> dict:
    confidence_payload = build_confidence_messages(row.get("confidence_label"))

    patient_why = build_patient_why_text(row)
    clinician_why = build_clinician_why_text(row)
    traceability_payload = build_traceability_payload(row)

    ml_rerank_note = _build_ml_rerank_note(row)
    safety_context_note = _build_safety_context_note(row)

    xai_payload = _build_bundle_xai_with_fallback(
        bundle_row=row,
        reranked_df=reranked_df,
        safety_context_df=safety_context_df,
        participant_features_df=participant_features_df,
    )
    xai_texts = build_ml_xai_texts(xai_payload)

    return {
        "patient_facing_explanation": patient_why,
        "clinician_facing_explanation": clinician_why,
        "patient_confidence_explanation": confidence_payload["patient_confidence_explanation"],
        "clinician_confidence_explanation": confidence_payload["clinician_confidence_explanation"],
        "safety_context_note": safety_context_note,
        "ml_rerank_note": ml_rerank_note,
        "traceability_payload": traceability_payload,
        "ml_xai_payload": xai_payload,
        "clinician_ml_xai_explanation": xai_texts["clinician_ml_xai_explanation"],
        "patient_ml_xai_explanation": xai_texts["patient_ml_xai_explanation"],
    }


def _build_bundle_xai_with_fallback(
    bundle_row: dict,
    reranked_df: pd.DataFrame | None,
    safety_context_df: pd.DataFrame | None,
    participant_features_df: pd.DataFrame | None,
) -> dict:
    try:
        payload = build_bundle_ml_xai_payload(
            bundle_row=bundle_row,
            reranked_df=reranked_df,
            safety_context_df=safety_context_df,
            participant_features_df=participant_features_df,
        )
    except Exception as exc:
        payload = {
            "xai_status": f"xai_failure:{type(exc).__name__}",
            "representative_action_code": None,
            "top_positive_factors": [],
            "top_negative_factors": [],
            "top_positive_factors_clinical_only": [],
            "top_negative_factors_clinical_only": [],
            "clinician_ml_xai_summary": "Local ML feature-attribution explanation was unavailable for this bundle.",
            "patient_ml_xai_summary": "A detailed model-based explanation was not available for this recommendation.",
        }

    if not isinstance(payload, dict):
        payload = {
            "xai_status": "xai_unavailable",
            "representative_action_code": None,
            "top_positive_factors": [],
            "top_negative_factors": [],
            "top_positive_factors_clinical_only": [],
            "top_negative_factors_clinical_only": [],
            "clinician_ml_xai_summary": "Local ML feature-attribution explanation was unavailable for this bundle.",
            "patient_ml_xai_summary": "A detailed model-based explanation was not available for this recommendation.",
        }

    xai_status = str(payload.get("xai_status", ""))
    failed = xai_status.startswith("xai_failure") or xai_status == "xai_unavailable"

    if failed:
        fallback_payload = _build_top_action_xai_fallback(
            bundle_row=bundle_row,
            reranked_df=reranked_df,
            safety_context_df=safety_context_df,
            participant_features_df=participant_features_df,
        )
        if fallback_payload:
            fallback_payload = fallback_payload.copy()
            fallback_payload["xai_status"] = f"bundle_xai_failed_action_fallback:{xai_status or 'unknown'}"
            return fallback_payload

    return payload


def _build_top_action_xai_fallback(
    bundle_row: dict,
    reranked_df: pd.DataFrame | None,
    safety_context_df: pd.DataFrame | None,
    participant_features_df: pd.DataFrame | None,
) -> dict:
    if reranked_df is None or reranked_df.empty:
        return {}

    participant_key = bundle_row.get("ParticipantKey")
    action_codes = bundle_row.get("action_codes", []) or []

    if not participant_key or not action_codes:
        return {}

    candidate_rows = reranked_df.loc[
        (reranked_df["ParticipantKey"].astype(str) == str(participant_key)) &
        (reranked_df["action_code"].astype(str).isin([str(a) for a in action_codes]))
    ].copy()

    if candidate_rows.empty:
        return {}

    sort_cols = [c for c in ["final_score", "total_score", "confidence_score", "priority"] if c in candidate_rows.columns]
    ascending = [False] * len(sort_cols)
    candidate_rows = candidate_rows.sort_values(by=sort_cols, ascending=ascending, kind="stable").reset_index(drop=True)

    representative_row = candidate_rows.iloc[0].to_dict()

    try:
        payload = build_bundle_ml_xai_payload(
            bundle_row=representative_row,
            reranked_df=reranked_df,
            safety_context_df=safety_context_df,
            participant_features_df=participant_features_df,
        )
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _build_ml_rerank_note(row) -> str:
    return (
        "Final bundle order may reflect bounded post-ranking refinement from the ML reranking layer, "
        "without overriding upstream clinical safety rules."
    )


def _build_safety_context_note(row) -> str:
    if bool(row.get("triage_mode", False)):
        return "Urgent safety context detected; escalation-focused communication was prioritised."
    return "No triage override detected; standard safety-constrained recommendation communication applied."
