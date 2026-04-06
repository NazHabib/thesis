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

    xai_payload = build_bundle_ml_xai_payload(
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


def _build_ml_rerank_note(row) -> str:
    return (
        "Final bundle order may reflect bounded post-ranking refinement from the ML reranking layer, "
        "without overriding upstream clinical safety rules."
    )


def _build_safety_context_note(row) -> str:
    if bool(row.get("triage_mode", False)):
        return "Urgent safety context detected; escalation-focused communication was prioritised."
    return "No triage override detected; standard safety-constrained recommendation communication applied."
