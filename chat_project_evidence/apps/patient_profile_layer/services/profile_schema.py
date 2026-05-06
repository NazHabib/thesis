from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from typing import Final, Iterable

import pandas as pd


PROFILE_VERSION_DEFAULT: Final[str] = "v1.0.0"

PROFILE_COLUMNS: Final[list[str]] = [
    "participant_key",
    "age_group",
    "cardiometabolic_risk",
    "glycaemic_status",
    "bp_status",
    "weight_status",
    "lifestyle_flags",
    "data_completeness_score",
    "profile_version",
    "created_at",
]

TEXT_COLUMNS: Final[list[str]] = [
    "participant_key",
    "age_group",
    "cardiometabolic_risk",
    "glycaemic_status",
    "bp_status",
    "weight_status",
    "lifestyle_flags",
    "profile_version",
]


@dataclass(frozen=True, slots=True)
class PatientProfileRecord:
    participant_key: str
    age_group: str
    cardiometabolic_risk: str
    glycaemic_status: str
    bp_status: str
    weight_status: str
    lifestyle_flags: str
    data_completeness_score: float
    profile_version: str
    created_at: datetime


def serialize_lifestyle_flags(flags: Iterable[str]) -> str:
    unique_flags = sorted({str(flag).strip() for flag in flags if str(flag).strip()})
    return json.dumps(unique_flags, ensure_ascii=False)


def deserialize_lifestyle_flags(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(item) for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            # fallback defensivo se o campo vier como "flag1|flag2"
            return [part for part in value.split("|") if part]
    return []


def empty_profile_df() -> pd.DataFrame:
    df = pd.DataFrame(columns=PROFILE_COLUMNS)
    return coerce_profile_dtypes(df)


def coerce_profile_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in TEXT_COLUMNS:
        if col in out.columns:
            out[col] = out[col].astype("string")

    if "data_completeness_score" in out.columns:
        out["data_completeness_score"] = (
            pd.to_numeric(out["data_completeness_score"], errors="coerce")
            .fillna(0.0)
            .clip(0.0, 1.0)
            .astype("float32")
        )

    if "created_at" in out.columns:
        out["created_at"] = pd.to_datetime(out["created_at"], utc=True, errors="coerce")

    missing_cols = [col for col in PROFILE_COLUMNS if col not in out.columns]
    for col in missing_cols:
        out[col] = pd.NA

    return out[PROFILE_COLUMNS]


def validate_profile_df(df: pd.DataFrame) -> pd.DataFrame:
    out = coerce_profile_dtypes(df)

    missing_cols = [col for col in PROFILE_COLUMNS if col not in out.columns]
    if missing_cols:
        raise ValueError(f"Missing required profile columns: {missing_cols}")

    if out["participant_key"].isna().any():
        raise ValueError("participant_key contains null values.")

    duplicated = out["participant_key"].astype(str).duplicated(keep=False)
    if duplicated.any():
        dupes = out.loc[duplicated, "participant_key"].astype(str).tolist()[:10]
        raise ValueError(f"Duplicate participant_key values detected: {dupes}")

    bad_scores = out["data_completeness_score"].dropna()
    if ((bad_scores < 0.0) | (bad_scores > 1.0)).any():
        raise ValueError("data_completeness_score must be within [0, 1].")

    # validação leve do payload serializado
    for raw_value in out["lifestyle_flags"].dropna().tolist():
        deserialize_lifestyle_flags(raw_value)

    return out
