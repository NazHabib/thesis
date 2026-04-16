from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


NON_FEATURE_COLUMNS = [
    "ParticipantKey",
    "label",
    "label_source",
    "training_row_type",
    "feature_schema_version",
    "label_schema_version",
    "rule_rank",
    "candidate_count_for_participant",
    "positive_count_for_participant",
]

CATEGORICAL_FEATURES_CANDIDATES = [
    "action_code",
    "action_type",
    "severity",
    "bundle_family",
    "confidence_label",
    "feature_tier",
    "confidence_ceiling_global",
    "coldstart_confidence_ceiling",
    "Severity_band",
]

NUMERIC_FEATURES_CANDIDATES = [
    "priority",
    "base_severity_score",
    "base_action_score",
    "evidence_strength",
    "domain_alignment",
    "missing_penalty",
    "contraindication_penalty",
    "total_score",
    "score_norm",
    "confidence_score",
    "data_completeness_score",
    "RIDAGEYR",
    "RIAGENDR",
    "BMXBMI",
    "BMXWAIST",
    "LBXGH",
    "SBP_mean",
    "DBP_mean",
    "Any_MVPA_flag",
    "Current_smoker_flag",
    "DietDay1_reliable_flag",
    "DR1TSODI",
    "DR1TSUGR",
    "DR1TFIBE",
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
    "HTN_severity_score",
    "DM_severity_score",
    "Obesity_severity_score",
    "Lifestyle_risk_score",
    "Cardiometabolic_severity_score",
    "is_diet_bundle",
    "is_movement_bundle",
    "is_weight_bundle",
    "is_smoking_bundle",
    "is_monitoring_bundle",
    "hba1c_high_flag",
    "hba1c_high_x_diet_bundle",
    "hba1c_high_x_monitoring_bundle",
    "sbp_high_flag",
    "sbp_high_x_movement_bundle",
    "sbp_high_x_monitoring_bundle",
    "obesity_flag_proxy",
    "bmi_x_weight_bundle",
    "smoker_x_smoking_bundle",
    "inactive_flag_proxy",
    "inactive_x_movement_bundle",
    "high_sodium_x_diet_bundle",
]

EXPERIMENT_C_EXCLUDE_FEATURES = {
    "score_norm",
    "total_score",
    "confidence_score",
    "priority",
    "base_severity_score",
    "base_action_score",
    "evidence_strength",
    "domain_alignment",
    "missing_penalty",
    "contraindication_penalty",
}


@dataclass
class PreparedFeatures:
    X_train: Any
    X_val: Any
    y_train: pd.Series
    y_val: pd.Series
    preprocessor: ColumnTransformer
    feature_columns: list[str]
    categorical_columns: list[str]
    numeric_columns: list[str]


def prepare_lgbm_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_col: str = "label",
    feature_profile: str = "default",
) -> PreparedFeatures:
    if target_col not in train_df.columns:
        raise ValueError(f"Missing target column in train_df: {target_col}")
    if target_col not in val_df.columns:
        raise ValueError(f"Missing target column in val_df: {target_col}")

    feature_columns = [
        c for c in train_df.columns
        if c not in NON_FEATURE_COLUMNS and c != target_col
    ]

    categorical_columns = [c for c in CATEGORICAL_FEATURES_CANDIDATES if c in feature_columns]
    numeric_columns = [c for c in NUMERIC_FEATURES_CANDIDATES if c in feature_columns]

    remaining = [c for c in feature_columns if c not in categorical_columns and c not in numeric_columns]
    inferred_numeric = []
    for col in remaining:
        if pd.api.types.is_numeric_dtype(train_df[col]):
            inferred_numeric.append(col)

    numeric_columns = _unique_preserve_order(numeric_columns + inferred_numeric)
    categorical_columns = _unique_preserve_order(categorical_columns)

    categorical_columns, numeric_columns = _apply_feature_profile(
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        feature_profile=feature_profile,
    )

    selected_columns = categorical_columns + numeric_columns

    X_train_df = train_df[selected_columns].copy()
    X_val_df = val_df[selected_columns].copy()

    y_train = train_df[target_col].astype(int).copy()
    y_val = val_df[target_col].astype(int).copy()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )

    X_train = preprocessor.fit_transform(X_train_df)
    X_val = preprocessor.transform(X_val_df)

    return PreparedFeatures(
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        preprocessor=preprocessor,
        feature_columns=selected_columns,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
    )


def _apply_feature_profile(
    categorical_columns: list[str],
    numeric_columns: list[str],
    feature_profile: str,
) -> tuple[list[str], list[str]]:
    profile = str(feature_profile).strip().lower()

    if profile == "default":
        return categorical_columns, numeric_columns

    if profile == "experiment_c":
        categorical_out = list(categorical_columns)
        numeric_out = [c for c in numeric_columns if c not in EXPERIMENT_C_EXCLUDE_FEATURES]
        return categorical_out, numeric_out

    raise ValueError(
        f"Unsupported feature_profile: {feature_profile}. "
        f"Supported values: default, experiment_c"
    )


def _unique_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
