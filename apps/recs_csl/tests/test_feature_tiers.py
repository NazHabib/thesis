import pandas as pd
from apps.recs_csl.services.cold_start.feature_tiers import build_feature_tiers


def test_feature_tier_full():
    df = pd.DataFrame([{
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

    out = build_feature_tiers(df)
    assert out.loc[0, 'feature_tier'] == 'full'
