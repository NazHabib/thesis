import pandas as pd

from apps.ranking_layer.services.ranking.arbitration import (
    apply_conflict_policy,
    apply_family_cap,
    arbitrate_candidates,
)


def _candidate(action_code, action_type="recommend", severity="moderate", priority=100, family="Monitoring", score=100):
    severity_score = {
        "low": 10,
        "moderate": 40,
        "high": 70,
        "critical": 100,
    }.get(severity, 0)

    return {
        "ParticipantKey": "P1",
        "action_code": action_code,
        "action_type": action_type,
        "severity": severity,
        "priority": priority,
        "priority_num": priority,
        "bundle_family": family,
        "total_score": score,
        "base_severity_score": severity_score,
    }


def test_conflict_policy_keep_strongest_keeps_top_candidate():
    df = pd.DataFrame([
        {
            **_candidate("LOW", severity="low", priority=100, score=50),
            "conflict_group": "G1",
            "conflict_policy": "keep_strongest",
        },
        {
            **_candidate("HIGH", severity="high", priority=200, score=90),
            "conflict_group": "G1",
            "conflict_policy": "keep_strongest",
        },
    ])

    out = apply_conflict_policy(df)
    assert out["action_code"].tolist() == ["HIGH"]


def test_family_cap_preserves_all_safety_items():
    rows = [
        _candidate("ESCALATE_A", action_type="escalate", severity="critical", priority=1100, family="Monitoring", score=200),
        _candidate("ESCALATE_B", action_type="escalate", severity="critical", priority=1050, family="Monitoring", score=190),
        _candidate("ESCALATE_C", action_type="escalate", severity="high", priority=900, family="Monitoring", score=180),
        _candidate("MONITOR_NORMAL_A", action_type="recommend", severity="moderate", priority=500, family="Monitoring", score=100),
        _candidate("MONITOR_NORMAL_B", action_type="recommend", severity="moderate", priority=400, family="Monitoring", score=90),
        _candidate("DIET_NORMAL", action_type="recommend", severity="moderate", priority=300, family="DietPattern", score=80),
    ]

    out = apply_family_cap(pd.DataFrame(rows))
    codes = set(out["action_code"].tolist())

    assert {"ESCALATE_A", "ESCALATE_B", "ESCALATE_C"}.issubset(codes)


def test_arbitrate_caps_normal_recommendations_per_participant():
    rows = [
        _candidate(f"DIET_{i}", family="DietPattern", score=100 - i)
        for i in range(5)
    ] + [
        _candidate(f"MOVE_{i}", family="Movement", score=90 - i)
        for i in range(5)
    ]

    out = arbitrate_candidates(pd.DataFrame(rows))

    assert len(out) <= 5
    assert (out.groupby("bundle_family").size() <= 2).all()
