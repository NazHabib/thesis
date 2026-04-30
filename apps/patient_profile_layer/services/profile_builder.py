from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .output_writer import write_profile_outputs
from .profile_schema import (
    PROFILE_VERSION_DEFAULT,
    empty_profile_df,
    serialize_lifestyle_flags,
    validate_profile_df,
)
from .snapshot import load_patient_profiles_snapshot


@dataclass(frozen=True, slots=True)
class ProfileColumnMap:
    participant_key: str = "ParticipantKey"
    age_years: str = "RIDAGEYR"
    bmi: str = "BMXBMI"
    waist: str | None = "BMXWAIST"
    systolic_bp: str = "SBP_mean"
    diastolic_bp: str = "DBP_mean"
    hba1c: str = "LBXGH"
    fasting_glucose_mg_dl: str | None = None

    known_diabetes_flag: str | None = "Diabetes"
    prediabetes_flag: str | None = "Prediabetes"
    current_smoker_flag: str | None = "Current_smoker_flag"
    alcohol_risk_flag: str | None = "Alcohol_over_guideline_flag"
    inactive_flag: str | None = "Sedentary_high_flag"
    poor_diet_flag: str | None = "Sodium_over_2300_flag"

    def all_input_columns(self) -> list[str]:
        cols = [
            self.participant_key,
            self.age_years,
            self.bmi,
            self.systolic_bp,
            self.diastolic_bp,
            self.hba1c,
            self.fasting_glucose_mg_dl,
            self.known_diabetes_flag,
            self.current_smoker_flag,
            self.alcohol_risk_flag,
            self.inactive_flag,
            self.poor_diet_flag,
        ]
        return [col for col in cols if col]


@dataclass(frozen=True, slots=True)
class ProfileBuildConfig:
    profile_version: str = PROFILE_VERSION_DEFAULT
    columns: ProfileColumnMap = ProfileColumnMap()


class PatientProfileBuilder:
    def __init__(self, config: ProfileBuildConfig | None = None) -> None:
        self.config = config or ProfileBuildConfig()
        self.columns = self.config.columns

    def build_from_frame(self, participant_df: pd.DataFrame) -> pd.DataFrame:
        if participant_df.empty:
            return empty_profile_df()

        if self.columns.participant_key not in participant_df.columns:
            raise ValueError(
                f"Required key column '{self.columns.participant_key}' not found."
            )

        df = participant_df.copy()
        participant_key = df[self.columns.participant_key].astype(str)

        age_years = self._numeric_column(df, self.columns.age_years)
        bmi = self._numeric_column(df, self.columns.bmi)
        sbp = self._numeric_column(df, self.columns.systolic_bp)
        dbp = self._numeric_column(df, self.columns.diastolic_bp)
        hba1c = self._numeric_column(df, self.columns.hba1c)
        fasting_glucose = self._numeric_column(df, self.columns.fasting_glucose_mg_dl)

        known_diabetes = self._bool_column(df, self.columns.known_diabetes_flag)
        prediabetes = self._bool_column(df, self.columns.prediabetes_flag)
        current_smoker = self._bool_column(df, self.columns.current_smoker_flag)
        alcohol_risk = self._bool_column(df, self.columns.alcohol_risk_flag)
        inactive = self._bool_column(df, self.columns.inactive_flag)
        poor_diet = self._bool_column(df, self.columns.poor_diet_flag)

        age_group = self._derive_age_group(age_years)
        weight_status = self._derive_weight_status(bmi)
        bp_status = self._derive_bp_status(sbp, dbp)
        glycaemic_status = self._derive_glycaemic_status(
            hba1c=hba1c,
            fasting_glucose=fasting_glucose,
            known_diabetes=known_diabetes,
            prediabetes=prediabetes,
        )
        lifestyle_flags = self._derive_lifestyle_flags(
            current_smoker=current_smoker,
            alcohol_risk=alcohol_risk,
            inactive=inactive,
            poor_diet=poor_diet,
        )
        data_completeness_score = self._derive_data_completeness_score(
            age_years=age_years,
            bmi=bmi,
            sbp=sbp,
            dbp=dbp,
            hba1c=hba1c,
            fasting_glucose=fasting_glucose,
            current_smoker=current_smoker,
            alcohol_risk=alcohol_risk,
            inactive=inactive,
            poor_diet=poor_diet,
        )
        cardiometabolic_risk = self._derive_cardiometabolic_risk(
            age_years=age_years,
            weight_status=weight_status,
            bp_status=bp_status,
            glycaemic_status=glycaemic_status,
            lifestyle_flags=lifestyle_flags,
            data_completeness_score=data_completeness_score,
        )

        out = pd.DataFrame(
            {
                "participant_key": participant_key,
                "age_group": age_group,
                "cardiometabolic_risk": cardiometabolic_risk,
                "glycaemic_status": glycaemic_status,
                "bp_status": bp_status,
                "weight_status": weight_status,
                "lifestyle_flags": lifestyle_flags,
                "data_completeness_score": data_completeness_score.round(3),
                "profile_version": self.config.profile_version,
                "created_at": pd.Timestamp.now(tz="UTC").floor("s"),
            }
        )

        out = out.drop_duplicates(subset=["participant_key"], keep="last")
        return validate_profile_df(out)

    @staticmethod
    def _numeric_column(df: pd.DataFrame, column_name: str | None) -> pd.Series:
        if not column_name or column_name not in df.columns:
            return pd.Series(np.nan, index=df.index, dtype="float64")
        return pd.to_numeric(df[column_name], errors="coerce")

    @staticmethod
    def _bool_column(df: pd.DataFrame, column_name: str | None) -> pd.Series:
        if not column_name or column_name not in df.columns:
            return pd.Series(pd.NA, index=df.index, dtype="boolean")

        series = df[column_name]

        if pd.api.types.is_bool_dtype(series):
            return series.fillna(False).astype("boolean")

        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().any():
            return (numeric.fillna(0) != 0).astype("boolean")

        normalized = series.astype("string").str.strip().str.lower()
        return normalized.isin({"1", "true", "t", "yes", "y"}).astype("boolean")

    @staticmethod
    def _derive_age_group(age_years: pd.Series) -> pd.Series:
        result = np.select(
            [
                age_years.between(18, 39, inclusive="both"),
                age_years.between(40, 64, inclusive="both"),
                age_years >= 65,
            ],
            [
                "18_39",
                "40_64",
                "65_plus",
            ],
            default="unknown",
        )
        return pd.Series(result, index=age_years.index, dtype="string")

    @staticmethod
    def _derive_weight_status(bmi: pd.Series) -> pd.Series:
        valid = bmi.where((bmi > 0) & (bmi < 100))
        result = np.select(
            [
                valid < 18.5,
                valid < 25.0,
                valid < 30.0,
                valid >= 30.0,
            ],
            [
                "underweight",
                "normal",
                "overweight",
                "obesity",
            ],
            default="unknown",
        )
        return pd.Series(result, index=bmi.index, dtype="string")

    @staticmethod
    def _derive_bp_status(sbp: pd.Series, dbp: pd.Series) -> pd.Series:
        result = np.select(
            [
                (sbp > 180) | (dbp > 120),
                (sbp >= 140) | (dbp >= 90),
                sbp.between(130, 139, inclusive="both") | dbp.between(80, 89, inclusive="both"),
                sbp.between(120, 129, inclusive="both") & (dbp < 80),
                (sbp < 120) & (dbp < 80),
            ],
            [
                "severe_hypertension",
                "stage_2_hypertension",
                "stage_1_hypertension",
                "elevated",
                "normal",
            ],
            default="unknown",
        )
        return pd.Series(result, index=sbp.index, dtype="string")

    @staticmethod
    def _derive_glycaemic_status(
        *,
        hba1c: pd.Series,
        fasting_glucose: pd.Series,
        known_diabetes: pd.Series,
        prediabetes: pd.Series,
    ) -> pd.Series:
        has_gly = hba1c.notna() | fasting_glucose.notna()

        above_target_known_diabetes = known_diabetes.fillna(False) & (hba1c >= 7.0)
        at_target_known_diabetes = known_diabetes.fillna(False) & hba1c.notna() & (hba1c < 7.0)

        diabetes_range = (hba1c >= 6.5) | (fasting_glucose >= 126)
        prediabetes_range = (
            prediabetes.fillna(False)
            | hba1c.between(5.7, 6.4, inclusive="both")
            | fasting_glucose.between(100, 125, inclusive="both")
        )
        normal = has_gly & ~(diabetes_range | prediabetes_range | known_diabetes.fillna(False))

        result = np.select(
            [
                above_target_known_diabetes,
                at_target_known_diabetes,
                diabetes_range,
                prediabetes_range,
                normal,
            ],
            [
                "above_target_known_diabetes",
                "at_target_known_diabetes",
                "diabetes_range",
                "prediabetes_range",
                "normal",
            ],
            default="unknown",
        )
        return pd.Series(result, index=hba1c.index, dtype="string")

    @staticmethod
    def _derive_lifestyle_flags(
        *,
        current_smoker: pd.Series,
        alcohol_risk: pd.Series,
        inactive: pd.Series,
        poor_diet: pd.Series,
    ) -> pd.Series:
        smoker_np = current_smoker.fillna(False).to_numpy(dtype=bool)
        alcohol_np = alcohol_risk.fillna(False).to_numpy(dtype=bool)
        inactive_np = inactive.fillna(False).to_numpy(dtype=bool)
        poor_diet_np = poor_diet.fillna(False).to_numpy(dtype=bool)

        payload = [
            serialize_lifestyle_flags(
                flag_name
                for flag_name, enabled in (
                    ("current_smoker", smoker),
                    ("high_alcohol_risk", alcohol),
                    ("inactive", inact),
                    ("poor_diet_signal", diet),
                )
                if enabled
            )
            for smoker, alcohol, inact, diet in zip(
                smoker_np,
                alcohol_np,
                inactive_np,
                poor_diet_np,
            )
        ]
        return pd.Series(payload, index=current_smoker.index, dtype="string")

    @staticmethod
    def _derive_data_completeness_score(
        *,
        age_years: pd.Series,
        bmi: pd.Series,
        sbp: pd.Series,
        dbp: pd.Series,
        hba1c: pd.Series,
        fasting_glucose: pd.Series,
        current_smoker: pd.Series,
        alcohol_risk: pd.Series,
        inactive: pd.Series,
        poor_diet: pd.Series,
    ) -> pd.Series:
        age_ok = age_years.notna()
        anthro_ok = bmi.notna()
        bp_ok = sbp.notna() & dbp.notna()
        gly_ok = hba1c.notna() | fasting_glucose.notna()
        lifestyle_ok = (
            current_smoker.notna()
            | alcohol_risk.notna()
            | inactive.notna()
            | poor_diet.notna()
        )

        score = (
            age_ok.astype("int8")
            + anthro_ok.astype("int8")
            + bp_ok.astype("int8")
            + gly_ok.astype("int8")
            + lifestyle_ok.astype("int8")
        ) / 5.0

        return score.astype("float32")

    @staticmethod
    def _derive_cardiometabolic_risk(
        *,
        age_years: pd.Series,
        weight_status: pd.Series,
        bp_status: pd.Series,
        glycaemic_status: pd.Series,
        lifestyle_flags: pd.Series,
        data_completeness_score: pd.Series,
    ) -> pd.Series:
        weight_points = weight_status.map(
            {
                "underweight": 0,
                "normal": 0,
                "overweight": 1,
                "obesity": 2,
            }
        ).fillna(0)

        bp_points = bp_status.map(
            {
                "normal": 0,
                "elevated": 1,
                "stage_1_hypertension": 2,
                "stage_2_hypertension": 3,
                "severe_hypertension": 4,
            }
        ).fillna(0)

        gly_points = glycaemic_status.map(
            {
                "normal": 0,
                "prediabetes_range": 1,
                "diabetes_range": 2,
                "at_target_known_diabetes": 2,
                "above_target_known_diabetes": 3,
            }
        ).fillna(0)

        age_points = (age_years >= 65).fillna(False).astype("int8")

        lifestyle_points = lifestyle_flags.map(
            lambda value: min(len(__import__("json").loads(value or "[]")), 2)
            if isinstance(value, str)
            else 0
        ).astype("int8")

        total_points = weight_points + bp_points + gly_points + age_points + lifestyle_points

        result = np.select(
            [
                data_completeness_score < 0.40,
                total_points >= 7,
                total_points >= 4,
                total_points >= 2,
            ],
            [
                "unknown",
                "very_high",
                "high",
                "moderate",
            ],
            default="low",
        )

        return pd.Series(result, index=weight_status.index, dtype="string")


def _available_parquet_columns(path: Path) -> list[str]:
    try:
        return pq.ParquetFile(path).schema.names
    except Exception:
        # fallback mais lento, mas robusto
        return list(pd.read_parquet(path).columns)


def build_profiles_from_path(
    *,
    participant_features_path: Path,
    output_jsonl_path: Path,
    output_parquet_path: Path,
    output_dataset_root: Path,
    profile_version: str = PROFILE_VERSION_DEFAULT,
    limit: int | None = None,
    incremental: bool = False,
    write_partitioned_dataset: bool = True,
    column_map: ProfileColumnMap | None = None,
) -> pd.DataFrame:
    columns = column_map or ProfileColumnMap()
    builder = PatientProfileBuilder(
        config=ProfileBuildConfig(
            profile_version=profile_version,
            columns=columns,
        )
    )

    available_columns = set(_available_parquet_columns(participant_features_path))
    columns_to_read = [
        col for col in columns.all_input_columns() if col in available_columns
    ]

    participant_df = pd.read_parquet(
        participant_features_path,
        columns=columns_to_read,
        engine="pyarrow",
    )

    participant_df[columns.participant_key] = participant_df[columns.participant_key].astype(str)

    if limit is not None:
        keys = (
            participant_df[columns.participant_key]
            .drop_duplicates()
            .head(int(limit))
            .tolist()
        )
        participant_df = participant_df[
            participant_df[columns.participant_key].isin(keys)
        ].copy()

    existing_df = pd.DataFrame()
    if incremental and output_parquet_path.exists():
        existing_df = load_patient_profiles_snapshot(output_parquet_path)
        if not existing_df.empty:
            existing_keys = set(existing_df["participant_key"].astype(str).tolist())
            participant_df = participant_df[
                ~participant_df[columns.participant_key].isin(existing_keys)
            ].copy()

    new_profiles_df = builder.build_from_frame(participant_df)

    if not existing_df.empty:
        final_df = pd.concat([existing_df, new_profiles_df], ignore_index=True)
        final_df = final_df.drop_duplicates(subset=["participant_key"], keep="last")
    else:
        final_df = new_profiles_df

    write_profile_outputs(
        final_df,
        jsonl_path=output_jsonl_path,
        parquet_path=output_parquet_path,
        dataset_root=output_dataset_root,
        write_partitioned_dataset=write_partitioned_dataset,
    )

    return validate_profile_df(final_df)
