import pandas as pd

from apps.ranking_layer.services.ranking.scoring import (
    count_passed_evidence,
    normalize_severity,
    score_candidates,
)


def _base_row(**overrides):
    row = {
        "ParticipantKey": "P1",
        "action_code": "ESCALATE_HTN_CRISIS",
        "action_type": "escalate",
        "severity": "critical",
        "priority": 1100,
        "tags": ["acute", "safety", "htn"],
        "evidence": [
            {"feature": "SBP_mean", "operator": ">=", "threshold": 180, "passed": True},
            {"feature": "DBP_mean", "operator": ">=", "threshold": 120, "passed": True},
        ],
        "source_rule_ids": ["HTNACUTE-0202"],
        "bundle_family": "Monitoring",
        "feature_tier": "full",
        "data_completeness_score": 1.0,
        "cold_start_flag": False,
        "triage_mode": True,
        "a1c_core": True,
        "bp_core": True,
        "bmi_core": True,
        "ceiling_DietPattern": "high",
        "ceiling_Movement": "medium",
        "ceiling_Smoking": "high",
        "ceiling_Monitoring": "high",
        "ceiling_Lipids": "medium",
        "ceiling_Alcohol": "medium",
        "Comorbidity": 0,
    }
    row.update(overrides)
    return row


def test_normalize_severity_supports_medium_alias():
    assert normalize_severity("medium") == "moderate"


def test_count_passed_evidence_deduplicates_same_atomic_condition():
    evidence = [
        {"feature": "SBP_mean", "operator": ">=", "threshold": 180, "passed": True},
        {"feature": "SBP_mean", "operator": ">=", "threshold": 180, "passed": True},
        {"feature": "DBP_mean", "operator": ">=", "threshold": 120, "passed": False},
    ]

    assert count_passed_evidence(evidence) == 1


def test_priority_scoring_preserves_high_safety_priority_range():
    df = pd.DataFrame([
        _base_row(priority=1100),
        _base_row(
            action_code="RECOMMEND_MVPA_START_GRADUAL",
            action_type="recommend",
            severity="moderate",
            priority=520,
            tags=["physical_activity", "mvpa"],
            evidence=[],
            source_rule_ids=["PA-0701"],
            bundle_family="Movement",
            triage_mode=False,
        ),
    ])

    scored = score_candidates(df)

    escalation_score = scored.loc[
        scored["action_code"] == "ESCALATE_HTN_CRISIS",
        "base_action_score",
    ].iloc[0]
    movement_score = scored.loc[
        scored["action_code"] == "RECOMMEND_MVPA_START_GRADUAL",
        "base_action_score",
    ].iloc[0]

    assert escalation_score > movement_score
    assert escalation_score >= 110


def test_missing_data_penalty_applies_to_minimal_cold_start():
    df = pd.DataFrame([
        _base_row(
            feature_tier="minimal",
            data_completeness_score=0.25,
            cold_start_flag=True,
        )
    ])

    scored = score_candidates(df)
    assert scored.loc[0, "missing_penalty"] > 0
