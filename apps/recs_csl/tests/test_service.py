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
