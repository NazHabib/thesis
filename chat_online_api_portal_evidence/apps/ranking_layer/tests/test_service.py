import pandas as pd

from apps.ranking_layer.services.ranking.service import run_ranking_layer


def _safety_context():
    return pd.DataFrame([
        {
            "ParticipantKey": "P1",
            "feature_tier": "full",
            "data_completeness_score": 1.0,
            "triage_mode": False,
            "confidence_ceiling_global": "high",
            "bp_core": True,
            "a1c_core": True,
            "bmi_core": True,
            "ceiling_DietPattern": "high",
            "ceiling_Movement": "medium",
            "ceiling_Weight": "high",
            "ceiling_Smoking": "high",
            "ceiling_Alcohol": "medium",
            "ceiling_Lipids": "medium",
            "ceiling_Monitoring": "high",
            "ceiling_IntegratedCardiometabolic": "high",
            "Comorbidity": 0,
        }
    ])


def _candidate(action_code, action_type="recommend", severity="moderate", priority=100, family="Movement"):
    return {
        "ParticipantKey": "P1",
        "action_code": action_code,
        "action_type": action_type,
        "severity": severity,
        "priority": priority,
        "tags": ["physical_activity"],
        "evidence": [],
        "source_rule_ids": ["R1"],
        "bundle_family": family,
        "coldstart_confidence_ceiling": "high",
        "coldstart_note_codes": [],
        "cold_start_flag": False,
    }


def test_ranking_layer_runs_and_outputs_required_columns():
    safety_context_df = _safety_context()
    gated_candidates_df = pd.DataFrame([
        _candidate("RECOMMEND_MVPA_START_GRADUAL", priority=520),
        _candidate("RECOMMEND_REDUCE_SEDENTARY_TIME", priority=400),
    ])

    result = run_ranking_layer(safety_context_df, gated_candidates_df)
    ranked = result["ranked_recommendations"]

    assert not ranked.empty
    assert "total_score" in ranked.columns
    assert "confidence_label" in ranked.columns
    assert "ranking_rationale_short" in ranked.columns
    assert result["manifest"]["ranking_version"].startswith("ranking_")


def test_ranking_layer_preserves_critical_escalation():
    safety_context_df = _safety_context()
    safety_context_df.loc[0, "triage_mode"] = True

    gated_candidates_df = pd.DataFrame([
        _candidate(
            "ESCALATE_HTN_CRISIS",
            action_type="escalate",
            severity="critical",
            priority=1100,
            family="Monitoring",
        ),
        _candidate(
            "RECOMMEND_MVPA_START_GRADUAL",
            action_type="recommend",
            severity="moderate",
            priority=520,
            family="Movement",
        ),
    ])

    result = run_ranking_layer(safety_context_df, gated_candidates_df)
    codes = result["ranked_recommendations"]["action_code"].tolist()

    assert "ESCALATE_HTN_CRISIS" in codes
