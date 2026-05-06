from __future__ import annotations


TRIAGE_REASON_PATIENT_TEXT = {
    "hypertensive_crisis": "very high blood pressure indicators",
    "severe_hyperglycemia": "very high blood sugar indicators",
    "both": "very high blood pressure and blood sugar indicators",
}

def build_patient_facing_explanation(
    bundle_title: str,
    rationale: str,
    confidence_label: str,
    triage_mode: bool,
    triage_reason: str,
    followup_timing_text: str | None = None,
) -> str:
    confidence_text = {
        "high": "Overall support for this recommendation is strong.",
        "medium": "Overall support for this recommendation is moderate.",
        "low": "Support for this recommendation is more limited, so it should be interpreted cautiously.",
    }.get(
        str(confidence_label).lower(),
        "This recommendation is based on the available information."
    )

    parts: list[str] = []

    if triage_mode:
        parts.append(
            f"{bundle_title} was prioritised because a higher-risk safety context was detected."
        )
    else:
        parts.append(f"{bundle_title} was prioritised based on the available clinical and lifestyle information.")

    if rationale:
        parts.append(str(rationale).strip())

    parts.append(confidence_text)

    if followup_timing_text:
        parts.append(str(followup_timing_text).strip())

    if triage_mode and triage_reason and str(triage_reason).lower() != "none":
        reason_text = TRIAGE_REASON_PATIENT_TEXT.get(
            str(triage_reason).lower(),
            "higher-risk clinical indicators",
        )
        parts.append(
            f"This was generated in safety-first mode because of {reason_text}."
        )

    return " ".join(parts)


def build_clinician_facing_explanation(
    bundle_family: str,
    action_count: int,
    confidence_label: str,
    bundle_score: float | int | None,
    source_rule_ids: list[str] | None,
    followup_urgency: str | None = None,
    followup_window_label: str | None = None,
    ml_rerank_note: str | None = None,
) -> str:
    rule_text = ", ".join(source_rule_ids or []) if source_rule_ids else "N/A"
    score_text = f"{float(bundle_score):.3f}" if bundle_score is not None else "N/A"

    parts = [
        f"Bundle family: {bundle_family}.",
        f"Aggregated action count: {action_count}.",
        f"Confidence label: {confidence_label}.",
        f"Bundle score: {score_text}.",
        f"Primary rule provenance: {rule_text}.",
    ]

    if followup_urgency:
        parts.append(f"Follow-up urgency: {followup_urgency}.")

    if followup_window_label:
        parts.append(f"Suggested follow-up window: {followup_window_label}.")

    if ml_rerank_note:
        parts.append(str(ml_rerank_note).strip())

    return " ".join(parts)


def build_patient_confidence_explanation(confidence_label: str) -> str:
    label = str(confidence_label).lower()

    if label == "high":
        return "The system found strong support for this recommendation."
    if label == "medium":
        return "The system found moderate support for this recommendation."
    if label == "low":
        return "The system found limited support for this recommendation, so it should be interpreted cautiously."

    return "Confidence is based on the available information."


def build_clinician_confidence_explanation(
    confidence_label: str,
    feature_tier: str | None = None,
    triage_mode: bool = False,
    coldstart_note_codes: list[str] | None = None,
) -> str:
    label = str(confidence_label).lower()
    tier_text = f"Feature tier: {feature_tier}. " if feature_tier else ""
    cold_text = ""
    if coldstart_note_codes:
        cold_text = f"Cold-start notes: {', '.join(coldstart_note_codes)}. "

    if triage_mode:
        return (
            f"{tier_text}{cold_text}"
            f"Confidence is {label}; active triage context with conservative communication and restricted ML influence."
        )

    if label == "high":
        return f"{tier_text}{cold_text}High confidence based on upstream rule strength, retained evidence, and available participant data."
    if label == "medium":
        return f"{tier_text}{cold_text}Moderate confidence; recommendation is supported but still constrained by available context."
    if label == "low":
        return f"{tier_text}{cold_text}Low confidence; recommendation should be interpreted cautiously because of informational uncertainty."

    return f"{tier_text}{cold_text}Confidence explanation based on the available context."


def build_safety_context_note(triage_mode: bool, triage_reason: str) -> str:
    if triage_mode:
        return (
            f"Triage context was detected ({triage_reason}); communication was adjusted to a more conservative safety-first mode."
        )
    return "No triage override was detected; standard safety-constrained recommendation communication was applied."


def build_ml_rerank_note(
    rerank_status: str | None,
    fallback_reason: str | None,
    fusion_strategy: str | None = None,
) -> str:
    status = str(rerank_status or "").lower()
    strategy = str(fusion_strategy or "").lower()

    if not status and not strategy:
        return "No ML reranking metadata was available for this recommendation."

    if status == "rules_only_fallback" or strategy == "rules_only_fallback":
        if fallback_reason:
            return f"Final ordering remained rules-only because the ML layer fell back ({fallback_reason})."
        return "Final ordering remained rules-only."

    strategy_text = f" Fusion strategy: {fusion_strategy}." if fusion_strategy else ""

    return (
        "Final ordering may reflect bounded ML reranking refinement without overriding upstream clinical safety constraints."
        f"{strategy_text}"
    )
