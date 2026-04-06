from rules_safety_engine.rules_engine.explain import (
    build_patient_explanation,
    build_clinician_explanation,
)
from rules_safety_engine.rules_engine.models import Recommendation, EvidenceItem


def test_patient_explanation():
    rec = Recommendation(
        participant_key="P1",
        action_code="REDUCE_SODIUM",
        action_type="recommend",
        severity="moderate",
        priority=50,
        title="Reduce Sodium",
        explanation="Your sodium intake is above 2300 mg/day.",
    )
    msg = build_patient_explanation(rec)
    assert "sodium" in msg.lower()


def test_clinician_explanation():
    rec = Recommendation(
        participant_key="P1",
        action_code="FOLLOWUP_HTN_STAGE2",
        action_type="recommend",
        severity="high",
        priority=100,
        title="Followup Htn Stage2",
        explanation="Stage 2 HTN range detected.",
        source_rule_ids=["HTN_STAGE2_FOLLOWUP"],
        evidence=[EvidenceItem(feature="SBP_mean", value=145)],
        tags=["htn"],
    )
    payload = build_clinician_explanation(rec)
    assert payload["participant_key"] == "P1"
    assert payload["source_rule_ids"] == ["HTN_STAGE2_FOLLOWUP"]
