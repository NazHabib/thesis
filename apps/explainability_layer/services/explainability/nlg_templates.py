from __future__ import annotations


def build_patient_facing_explanation(
    bundle_title: str,
    rationale: str,
    confidence_label: str,
    triage_mode: bool,
    triage_reason: str,
) -> str:
    confidence_text = {
        "high": "This recommendation is supported by strong available evidence.",
        "medium": "This recommendation is supported by moderate available evidence.",
        "low": "This recommendation should be interpreted more cautiously because the available support is limited.",
    }.get(str(confidence_label).lower(), "This recommendation is based on the available information.")

    if triage_mode:
        return (
            f"{bundle_title} was prioritised under a higher-safety clinical context. "
            f"{rationale} "
            f"The system is operating in safety-first mode ({triage_reason}), so this recommendation should be interpreted with additional caution. "
            f"{confidence_text}"
        )

    return (
        f"{bundle_title} was selected to support safer self-management. "
        f"{rationale} "
        f"{confidence_text}"
    )


def build_clinician_facing_explanation(
    bundle_family: str,
    action_count: int,
    confidence_label: str,
    bundle_score: float | int | None,
    source_rule_ids: list[str] | None,
    ml_rerank_note: str | None = None,
) -> str:
    rule_text = ", ".join(source_rule_ids or []) if source_rule_ids else "N/A"
    score_text = f"{float(bundle_score):.3f}" if bundle_score is not None else "N/A"

    base = (
        f"Bundle family: {bundle_family}. "
        f"Aggregated action count: {action_count}. "
        f"Confidence label: {confidence_label}. "
        f"Bundle score: {score_text}. "
        f"Primary rule provenance: {rule_text}."
    )

    if ml_rerank_note:
        base = f"{base} {ml_rerank_note}"

    return base


def build_patient_confidence_explanation(confidence_label: str) -> str:
    label = str(confidence_label).lower()

    if label == "high":
        return "The system has strong supporting evidence for this recommendation."
    if label == "medium":
        return "The system has moderate supporting evidence for this recommendation."
    if label == "low":
        return "The system has limited supporting evidence for this recommendation, so it should be interpreted cautiously."

    return "The confidence in this recommendation is based on the available information."


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
        return f"{tier_text}{cold_text}Moderate confidence; recommendation is supported but subject to data limitations or weaker contextual support."
    if label == "low":
        return f"{tier_text}{cold_text}Low confidence; recommendation should be interpreted cautiously due to informational uncertainty."

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

    if status == "rules_only_fallback":
        if fallback_reason:
            return (
                f"Final ordering remained rules-only because the ML layer fell back ({fallback_reason})."
            )
        return "Final ordering remained rules-only."

    strategy_text = ""
    if fusion_strategy:
        strategy_text = f" Fusion strategy: {fusion_strategy}."

    return (
        "Final ordering may reflect bounded ML reranking refinement without overriding upstream clinical safety constraints."
        f"{strategy_text}"
    )
