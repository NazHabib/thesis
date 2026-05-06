import pandas as pd

from apps.bundling_layer.services.bundling.bundler import build_bundles


def test_triage_keeps_escalation():
    df = pd.DataFrame([
        {
            "ParticipantKey": "P1",
            "bundle_family": "Monitoring",
            "action_code": "ESCALATE_HTN_CRISIS",
            "action_type": "escalate",
            "priority": 100,
            "tags": ["acute"],
            "source_rule_ids": ["R1"],
            "evidence": [],
            "coldstart_note_codes": [],
            "total_score": 200.0,
            "confidence_score": 1.0,
            "confidence_label": "high",
            "triage_mode": True,
            "triage_reason": "hypertensive_crisis",
        },
        {
            "ParticipantKey": "P1",
            "bundle_family": "DietPattern",
            "action_code": "RECOMMEND_DASH",
            "action_type": "recommend",
            "priority": 500,
            "tags": ["diet"],
            "source_rule_ids": ["R2"],
            "evidence": [],
            "coldstart_note_codes": [],
            "total_score": 150.0,
            "confidence_score": 0.8,
            "confidence_label": "medium",
            "triage_mode": True,
            "triage_reason": "hypertensive_crisis",
        },
    ])

    out = build_bundles(df)
    assert len(out) == 1
    assert out.iloc[0]["action_type"] == "escalate"


def test_diet_items_merge_into_one_bundle():
    df = pd.DataFrame([
        {
            "ParticipantKey": "P2",
            "bundle_family": "DietPattern",
            "action_code": "RECOMMEND_DASH_OR_MEDITERRANEAN_PATTERN",
            "action_type": "recommend",
            "priority": 580,
            "tags": ["diet", "pattern"],
            "source_rule_ids": ["R1"],
            "evidence": [],
            "coldstart_note_codes": [],
            "total_score": 150.0,
            "confidence_score": 0.9,
            "confidence_label": "high",
            "triage_mode": False,
            "triage_reason": "none",
        },
        {
            "ParticipantKey": "P2",
            "bundle_family": "DietPattern",
            "action_code": "INFORM_VERY_HIGH_SODIUM_DAY1",
            "action_type": "inform",
            "priority": 680,
            "tags": ["diet", "sodium"],
            "source_rule_ids": ["R2"],
            "evidence": [],
            "coldstart_note_codes": [],
            "total_score": 140.0,
            "confidence_score": 0.8,
            "confidence_label": "medium",
            "triage_mode": False,
            "triage_reason": "none",
        },
    ])

    out = build_bundles(df)
    assert len(out) == 1
    assert out.iloc[0]["bundle_family"] == "DietPattern"
    assert len(out.iloc[0]["action_codes"]) == 2

def test_triage_tie_break_prefers_higher_priority_escalation():
    df = pd.DataFrame([
        {
            "ParticipantKey": "P1",
            "bundle_family": "Monitoring",
            "action_code": "ESCALATE_LOW_PRIORITY",
            "action_type": "escalate",
            "priority": 100,
            "tags": ["acute"],
            "source_rule_ids": ["R1"],
            "evidence": [],
            "coldstart_note_codes": [],
            "total_score": 200.0,
            "confidence_score": 1.0,
            "confidence_label": "high",
            "triage_mode": True,
            "triage_reason": "hypertensive_crisis",
            "followup_window_order": 1,
            "followup_horizon_days": 0,
        },
        {
            "ParticipantKey": "P1",
            "bundle_family": "Monitoring",
            "action_code": "ESCALATE_HIGH_PRIORITY",
            "action_type": "escalate",
            "priority": 1100,
            "tags": ["acute"],
            "source_rule_ids": ["R2"],
            "evidence": [],
            "coldstart_note_codes": [],
            "total_score": 200.0,
            "confidence_score": 1.0,
            "confidence_label": "high",
            "triage_mode": True,
            "triage_reason": "hypertensive_crisis",
            "followup_window_order": 1,
            "followup_horizon_days": 0,
        },
    ])

    out = build_bundles(df)

    assert len(out) == 1
    assert out.iloc[0]["action_type"] == "escalate"
    assert "ESCALATE_HIGH_PRIORITY" in out.iloc[0]["action_codes"]


def test_system_scope_message_removed_from_patient_bundles():
    df = pd.DataFrame([
        {
            "ParticipantKey": "P1",
            "bundle_family": "Monitoring",
            "action_code": "INFORM_SYSTEM_SCOPE",
            "action_type": "inform",
            "priority": 10,
            "tags": ["system"],
            "source_rule_ids": ["SAFETY-0001"],
            "evidence": [],
            "coldstart_note_codes": [],
            "total_score": 20.0,
            "confidence_score": 0.5,
            "confidence_label": "medium",
            "triage_mode": False,
            "triage_reason": "none",
        },
        {
            "ParticipantKey": "P1",
            "bundle_family": "Movement",
            "action_code": "RECOMMEND_MVPA_START_GRADUAL",
            "action_type": "recommend",
            "priority": 520,
            "tags": ["physical_activity", "mvpa"],
            "source_rule_ids": ["PA-0701"],
            "evidence": [],
            "coldstart_note_codes": [],
            "total_score": 130.0,
            "confidence_score": 0.8,
            "confidence_label": "medium",
            "triage_mode": False,
            "triage_reason": "none",
        },
    ])

    out = build_bundles(df)

    all_codes = []
    for codes in out["action_codes"]:
        all_codes.extend(codes)

    assert "INFORM_SYSTEM_SCOPE" not in all_codes
    assert "RECOMMEND_MVPA_START_GRADUAL" in all_codes
