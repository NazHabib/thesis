from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .models import EvidenceItem, FiredRule, Recommendation


def evidence_item_to_text(e: EvidenceItem) -> str:
    if e.operator is not None and e.threshold is not None:
        return f"{e.feature}={e.value} ({e.operator} {e.threshold})"
    return f"{e.feature}={e.value}"


def summarize_evidence(evidence: Sequence[EvidenceItem], max_items: int = 5) -> List[str]:
    items = [evidence_item_to_text(e) for e in evidence[:max_items]]
    return items


def build_patient_explanation(recommendation: Recommendation) -> str:
    """
    Patient-facing explanation.
    Keeps the language simple and avoids technical overload.
    """
    if recommendation.explanation:
        return recommendation.explanation

    title = recommendation.title or recommendation.action_code.replace("_", " ").title()

    if recommendation.action_type == "escalate":
        return f"{title}. This should be reviewed urgently by a healthcare professional."
    if recommendation.action_type == "forbid":
        return f"{title}. This is being avoided for safety reasons."
    if recommendation.action_type == "recommend":
        return f"{title}. This recommendation was generated from your current health profile."
    return f"{title}."


def build_clinician_explanation(recommendation: Recommendation) -> Dict[str, Any]:
    """
    Clinician-facing structured explanation.
    """
    return {
        "participant_key": recommendation.participant_key,
        "action_code": recommendation.action_code,
        "action_type": recommendation.action_type,
        "severity": recommendation.severity,
        "priority": recommendation.priority,
        "source_rule_ids": list(recommendation.source_rule_ids),
        "tags": list(recommendation.tags),
        "evidence_summary": summarize_evidence(recommendation.evidence),
        "full_evidence": [
            {
                "feature": e.feature,
                "value": e.value,
                "operator": e.operator,
                "threshold": e.threshold,
                "passed": e.passed,
            }
            for e in recommendation.evidence
        ],
        "raw_explanation": recommendation.explanation,
    }


def build_fired_rule_explanation(rule: FiredRule) -> Dict[str, Any]:
    return {
        "participant_key": rule.participant_key,
        "rule_id": rule.rule_id,
        "rule_version": rule.rule_version,
        "action_code": rule.action_code,
        "action_type": rule.action_type,
        "severity": rule.severity,
        "priority": rule.priority,
        "group": rule.group,
        "tags": list(rule.tags),
        "cycle": rule.cycle,
        "explanation": rule.explanation,
        "evidence_summary": summarize_evidence(rule.evidence),
        "full_evidence": [
            {
                "feature": e.feature,
                "value": e.value,
                "operator": e.operator,
                "threshold": e.threshold,
                "passed": e.passed,
            }
            for e in rule.evidence
        ],
    }


def explain_recommendation_bundle(
    recommendations: Sequence[Recommendation],
    fired_rules: Optional[Sequence[FiredRule]] = None,
) -> Dict[str, Any]:
    """
    Bundle-style explanation output for UI/API.
    """
    patient_view = [
        {
            "participant_key": r.participant_key,
            "action_code": r.action_code,
            "title": r.title,
            "message": build_patient_explanation(r),
            "severity": r.severity,
            "action_type": r.action_type,
        }
        for r in recommendations
    ]

    clinician_view = [build_clinician_explanation(r) for r in recommendations]

    payload: Dict[str, Any] = {
        "patient_view": patient_view,
        "clinician_view": clinician_view,
    }

    if fired_rules is not None:
        payload["fired_rules"] = [build_fired_rule_explanation(r) for r in fired_rules]

    return payload
