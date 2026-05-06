import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, Normalizer


NUMERIC_COLUMNS = [
    "RIDAGEYR",
    "BMXBMI",
    "BMXWAIST",
    "LBXGH",
    "SBP_mean",
    "DBP_mean",
    "DR1TSODI",
    "DR1TSUGR",
    "DR1TFIBE",
    "data_completeness_score",
    "followup_horizon_days",
    "persistence_risk_proxy",
    "HTN_severity_score",
    "DM_severity_score",
    "Obesity_severity_score",
    "Lifestyle_risk_score",
    "Cardiometabolic_severity_score",
]

BINARY_COLUMNS = [
    "Any_MVPA_flag",
    "Current_smoker_flag",
    "DietDay1_reliable_flag",
    "Alcohol_over_guideline_flag",
    "Hypertension",
    "Hypertensive_crisis_flag",
    "Diabetes",
    "Prediabetes",
    "Severe_hyperglycemia_flag",
    "Severe_obesity_flag",
    "Sedentary_high_flag",
    "Sodium_over_2300_flag",
    "Sodium_over_1500_flag",
    "SatFat_over_10pct_flag",
    "HDL_low_flag",
    "TC_high_flag",
    "Comorbidity",
    "repeat_measurement_available",
    "longitudinal_ready_flag",
]

CATEGORICAL_COLUMNS = [
    "RIAGENDR",
    "feature_tier",
    "followup_urgency",
    "monitoring_need_level",
    "temporal_risk_tier",
    "temporal_data_quality",
]


def available_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [c for c in columns if c in df.columns]


def build_similarity_preprocessor(df: pd.DataFrame) -> Pipeline:
    numeric_cols = available_columns(df, NUMERIC_COLUMNS)
    binary_cols = available_columns(df, BINARY_COLUMNS)
    categorical_cols = available_columns(df, CATEGORICAL_COLUMNS)

    transformers = []

    if numeric_cols:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric_cols,
        ))

    if binary_cols:
        transformers.append((
            "bin",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
            ]),
            binary_cols,
        ))

    if categorical_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_cols,
        ))

    return Pipeline([
        ("columns", ColumnTransformer(transformers=transformers, remainder="drop")),
        ("normalizer", Normalizer(norm="l2")),
    ])
