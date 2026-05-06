from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import pandas as pd
from django.conf import settings

from apps.accounts.models import User
from apps.profiles.services import build_current_health_feature_payload


@dataclass
class Rec:
    title: str
    category: str
    priority: int
    rationale: str
    explanation: str
    source: str
    confidence: float
    reference_participant_key: str = ""
    reference_match_distance: float | None = None
    matched_feature_count: int = 0


MATCHING_FEATURES = [
    "RIDAGEYR",
    "RIAGENDR",
    "BMXBMI",
    "BMXWAIST",
    "LBXGH",
    "SBP_mean",
    "DBP_mean",
    "DR1TSODI",
    "DR1TSUGR",
    "DR1TFIBE",
    "Any_MVPA_flag",
    "Current_smoker_flag",
    "Alcohol_over_guideline_flag",
    "Sedentary_high_flag",
    "TotalChol_mg_dL",
    "HDL_mg_dL",
    "Hypertension",
    "Diabetes",
    "Prediabetes",
    "Severe_hyperglycemia_flag",
    "Hypertensive_crisis_flag",
    "Severe_obesity_flag",
    "Sodium_over_2300_flag",
    "Sodium_over_1500_flag",
    "HDL_low_flag",
    "TC_high_flag",
    "Comorbidity",
    "HTN_severity_score",
    "DM_severity_score",
    "Obesity_severity_score",
    "Lifestyle_risk_score",
    "Cardiometabolic_severity_score",
]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    return rows


def _load_engineered_dataset() -> pd.DataFrame:
    base = Path(settings.BASE_DIR)

    parquet_path = base / "nhanes_step1_out" / "NHANES_Master_engineered.parquet"
    csv_path = base / "nhanes_step1_out" / "NHANES_Master_engineered.csv"

    if parquet_path.exists():
        return pd.read_parquet(parquet_path)

    if csv_path.exists():
        return pd.read_csv(csv_path)

    return pd.DataFrame()


def _safe_float(value: Any) -> float | None:
    if value in [None, ""]:
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if math.isnan(number):
        return None

    return number


def _available_recommendation_participant_keys(rows: list[dict[str, Any]]) -> set[str]:
    return {
        str(row.get("ParticipantKey"))
        for row in rows
        if row.get("ParticipantKey") not in [None, ""]
    }


def _find_closest_reference_participant(
    user: User,
    available_participant_keys: set[str],
) -> tuple[str | None, float | None, int, dict[str, Any]]:
    """
    Match the user's current health feature payload to the closest participant
    that also has audited explained recommendations available.

    This prevents selecting a reference participant for whom no recommendation
    output exists.
    """
    user_payload = build_current_health_feature_payload(user)

    clean_user_payload: dict[str, float] = {}
    for key, value in user_payload.items():
        if key.startswith("_"):
            continue

        number = _safe_float(value)
        if number is not None:
            clean_user_payload[key] = number

    if not clean_user_payload:
        return None, None, 0, user_payload

    df = _load_engineered_dataset()

    if df.empty or "ParticipantKey" not in df.columns:
        return None, None, 0, user_payload

    df = df.copy()
    df["ParticipantKey"] = df["ParticipantKey"].astype(str)

    if available_participant_keys:
        df = df[df["ParticipantKey"].isin(available_participant_keys)]

    if df.empty:
        return None, None, 0, user_payload

    usable_features = [
        feature
        for feature in MATCHING_FEATURES
        if feature in df.columns and feature in clean_user_payload
    ]

    if not usable_features:
        return None, None, 0, user_payload

    match_df = df[["ParticipantKey"] + usable_features].copy()

    for feature in usable_features:
        match_df[feature] = pd.to_numeric(match_df[feature], errors="coerce")

    distance = pd.Series(0.0, index=match_df.index)
    used_count = 0

    for feature in usable_features:
        user_value = clean_user_payload.get(feature)

        if user_value is None:
            continue

        col = match_df[feature]
        std = col.std(skipna=True)

        if not std or pd.isna(std) or std == 0:
            std = 1.0

        feature_distance = ((col - user_value) / std) ** 2

        # If a dataset participant is missing this feature, avoid letting that
        # feature dominate the distance calculation.
        feature_distance = feature_distance.fillna(0.0)

        distance = distance + feature_distance
        used_count += 1

    if used_count == 0:
        return None, None, 0, user_payload

    match_df["_distance"] = (distance / used_count) ** 0.5
    best = match_df.sort_values("_distance", ascending=True).iloc[0]

    return (
        str(best["ParticipantKey"]),
        float(best["_distance"]),
        used_count,
        user_payload,
    )


def _patient_priority_from_rank(rank: int) -> int:
    if rank <= 1:
        return 1
    if rank <= 3:
        return 2
    if rank <= 5:
        return 3
    return 4


def _safe_text(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback

    text = str(value).strip()
    return text if text else fallback


def _select_participant_rows(
    rows: list[dict[str, Any]],
    participant_key: str | None,
) -> list[dict[str, Any]]:
    if not rows:
        return []

    if participant_key:
        matched_rows = [
            row
            for row in rows
            if str(row.get("ParticipantKey")) == str(participant_key)
        ]

        if matched_rows:
            return matched_rows

    first_participant = rows[0].get("ParticipantKey")
    return [
        row
        for row in rows
        if row.get("ParticipantKey") == first_participant
    ]


def generate_recommendations(user_context: dict | None = None) -> List[Rec]:
    """
    Patient-facing recommendation adapter.

    It uses the user's saved health record timeline to choose a closest
    reference participant from the engineered dataset, then returns that
    participant's already-audited hybrid recommendations.

    The patient UI receives only patient-safe recommendation fields.
    """
    user_context = user_context or {}

    output_path = (
        Path(settings.BASE_DIR)
        / "apps"
        / "ml_layer"
        / "output"
        / "explained_recommendations.jsonl"
    )

    rows = _load_jsonl(output_path)

    if not rows:
        return []

    available_keys = _available_recommendation_participant_keys(rows)

    reference_participant_key = user_context.get("participant_key")
    reference_distance = None
    matched_feature_count = 0

    user_id = user_context.get("user_id")

    if not reference_participant_key and user_id:
        try:
            user = User.objects.get(id=user_id)
            (
                reference_participant_key,
                reference_distance,
                matched_feature_count,
                _user_payload,
            ) = _find_closest_reference_participant(user, available_keys)
        except User.DoesNotExist:
            reference_participant_key = None

    participant_rows = _select_participant_rows(rows, reference_participant_key)

    participant_rows = sorted(
        participant_rows,
        key=lambda row: (
            int(row.get("followup_window_order") or 99),
            -float(row.get("bundle_score") or 0),
            -float(row.get("priority") or 0),
        ),
    )

    recs: list[Rec] = []

    for rank, row in enumerate(participant_rows[:5], start=1):
        title = _safe_text(row.get("bundle_title"), "Health recommendation")
        category = _safe_text(row.get("category"), "guidance")

        rationale = _safe_text(
            row.get("rationale"),
            "This recommendation was selected based on the available health information.",
        )

        patient_explanation = _safe_text(row.get("patient_facing_explanation"))
        followup_text = _safe_text(row.get("followup_timing_text"))
        confidence_text = _safe_text(row.get("patient_confidence_explanation"))

        explanation_parts = []

        if patient_explanation:
            explanation_parts.append(patient_explanation)

        if confidence_text and confidence_text not in patient_explanation:
            explanation_parts.append(confidence_text)

        if followup_text and followup_text not in patient_explanation:
            explanation_parts.append(followup_text)

        explanation = " ".join(explanation_parts).strip()

        if not explanation:
            explanation = (
                "This recommendation may help guide your next care steps. "
                "Please discuss it with your healthcare professional if you are unsure."
            )

        recs.append(
            Rec(
                title=title,
                category=category,
                priority=_patient_priority_from_rank(rank),
                rationale=rationale,
                explanation=explanation,
                source="reference_match",
                confidence=float(row.get("confidence_score") or 0.0),
                reference_participant_key=str(row.get("ParticipantKey") or reference_participant_key or ""),
                reference_match_distance=reference_distance,
                matched_feature_count=matched_feature_count,
            )
        )

    return recs
