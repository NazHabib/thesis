import pandas as pd
from apps.recs_csl.services.cold_start.service import run_cold_start_layer


def test_service_runs():
    cfm_df = pd.DataFrame([{
        'ParticipantKey': 'P1',
        'Cycle': '2017-2018',
        'SBP_mean': 130,
        'DBP_mean': 80,
        'BP_n_used': 2,
        'LBXGH': 6.0,
        'BMXBMI': 31,
        'DietDay1_reliable_flag': 1,
        'DR1TSODI': 2500,
        'Current_smoker_flag': 1,
        'Any_MVPA_flag': 0,
        'Hypertensive_crisis_flag': 0,
        'Severe_hyperglycemia_flag': 0,
    }])

    participants_df = pd.DataFrame([{
        'ParticipantKey': 'P1',
        'Candidates': [{
            'action_code': 'MOVE_001',
            'action_type': 'recommend',
            'severity': 'moderate',
            'priority': 50,
            'tags': ['Movement'],
            'evidence': [],
            'source_rule_ids': ['R1'],
            'bundle_family': 'Movement',
        }]
    }])

    result = run_cold_start_layer(cfm_df, participants_df)
    assert 'safety_context' in result
    assert 'gated_candidates' in result
    
def test_triage_drops_non_escalate_and_keeps_escalate():
    cfm_df = pd.DataFrame([{
        "ParticipantKey": "P1",
        "Cycle": "2017-2018",
        "SBP_mean": 185,
        "DBP_mean": 121,
        "BP_n_used": 2,
        "LBXGH": 6.0,
        "BMXBMI": 31,
        "DietDay1_reliable_flag": 1,
        "DR1TSODI": 2500,
        "Current_smoker_flag": 0,
        "Any_MVPA_flag": 0,
        "Hypertensive_crisis_flag": 1,
        "Severe_hyperglycemia_flag": 0,
    }])

    participants_df = pd.DataFrame([{
        "ParticipantKey": "P1",
        "Candidates": [
            {
                "action_code": "ESCALATE_HTN_CRISIS",
                "action_type": "escalate",
                "severity": "critical",
                "priority": 1050,
                "tags": ["acute", "htn"],
                "evidence": [],
                "source_rule_ids": ["HTNACUTE-0202"],
                "bundle_family": "Monitoring",
            },
            {
                "action_code": "RECOMMEND_MVPA_START_GRADUAL",
                "action_type": "recommend",
                "severity": "moderate",
                "priority": 520,
                "tags": ["physical_activity"],
                "evidence": [],
                "source_rule_ids": ["PA-0701"],
                "bundle_family": "Movement",
            },
        ],
    }])

    result = run_cold_start_layer(cfm_df, participants_df)
    codes = result["gated_candidates"]["action_code"].tolist()

    assert "ESCALATE_HTN_CRISIS" in codes
    assert "RECOMMEND_MVPA_START_GRADUAL" not in codes

def test_triage_with_no_candidates_generates_fallback_escalation():
    cfm_df = pd.DataFrame([{
        "ParticipantKey": "P1",
        "Cycle": "2017-2018",
        "SBP_mean": 185,
        "DBP_mean": 121,
        "BP_n_used": 2,
        "LBXGH": 6.0,
        "BMXBMI": 31,
        "DietDay1_reliable_flag": 1,
        "DR1TSODI": 2500,
        "Current_smoker_flag": 0,
        "Any_MVPA_flag": 0,
        "Hypertensive_crisis_flag": 1,
        "Severe_hyperglycemia_flag": 0,
    }])

    participants_df = pd.DataFrame([{
        "ParticipantKey": "P1",
        "Candidates": [],
    }])

    result = run_cold_start_layer(cfm_df, participants_df)
    gated = result["gated_candidates"]

    assert not gated.empty
    assert "CSL_ESCALATE_TRIAGE_NO_RULE" in gated["action_code"].tolist()

    fallback = gated.loc[gated["action_code"] == "CSL_ESCALATE_TRIAGE_NO_RULE"].iloc[0]
    assert fallback["action_type"] == "escalate"
    assert fallback["severity"] == "critical"
    assert fallback["coldstart_confidence_ceiling"] == "high"


