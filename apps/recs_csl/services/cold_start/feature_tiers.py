import numpy as np
import pandas as pd

from .config import (
    FEATURE_TIER_MINIMAL,
    FEATURE_TIER_CORE,
    FEATURE_TIER_FULL,
    MIN_BP_READINGS_FOR_CORE,
)


def build_feature_tiers(cfm_df: pd.DataFrame) -> pd.DataFrame:
    df = cfm_df.copy()

    bp_present = df['SBP_mean'].notna() & df['DBP_mean'].notna()
    bp_n_ok = df['BP_n_used'].fillna(0).astype('int16') >= MIN_BP_READINGS_FOR_CORE

    df['bp_core'] = bp_present & bp_n_ok
    df['a1c_core'] = df['LBXGH'].notna()
    df['bmi_core'] = df['BMXBMI'].notna()
    df['diet_core'] = (
        df['DietDay1_reliable_flag'].fillna(0).astype('int8').eq(1)
        & df['DR1TSODI'].notna()
    )

    core_ready = df['bp_core'] & df['a1c_core'] & df['bmi_core']
    full_ready = core_ready & df['diet_core']

    df['feature_tier'] = np.select(
        [full_ready, core_ready],
        [FEATURE_TIER_FULL, FEATURE_TIER_CORE],
        default=FEATURE_TIER_MINIMAL,
    )

    df['data_completeness_score'] = (
        df[['bp_core', 'a1c_core', 'bmi_core', 'diet_core']]
        .astype('int8')
        .sum(axis=1)
        .div(4.0)
        .round(3)
    )

    return df
