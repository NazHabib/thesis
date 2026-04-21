from __future__ import annotations

import pandas as pd


FOLLOWUP_WINDOW_ORDER = {
    "within_1_month": 1,
    "within_3_months": 2,
    "within_6_months": 3,
    "routine_annual": 4,
    "unknown": 99,
}


def derive_followup_window_label(
    followup_horizon_days,
    followup_urgency: str | None = None,
) -> str:
    horizon = pd.to_numeric(pd.Series([followup_horizon_days]), errors="coerce").iloc[0]
    urgency = str(followup_urgency).strip().lower() if followup_urgency is not None else ""

    if pd.isna(horizon):
        if urgency in {"urgent", "soon", "monitor", "routine"}:
            return {
                "urgent": "within_1_month",
                "soon": "within_3_months",
                "monitor": "within_6_months",
                "routine": "routine_annual",
            }[urgency]
        return "unknown"

    if horizon <= 30:
        return "within_1_month"
    if horizon <= 90:
        return "within_3_months"
    if horizon <= 180:
        return "within_6_months"
    if horizon <= 365:
        return "routine_annual"

    if urgency in {"urgent", "soon", "monitor", "routine"}:
        return {
            "urgent": "within_1_month",
            "soon": "within_3_months",
            "monitor": "within_6_months",
            "routine": "routine_annual",
        }.get(urgency, "unknown")

    return "unknown"


def derive_followup_window_order(followup_window_label: str | None) -> int:
    return FOLLOWUP_WINDOW_ORDER.get(str(followup_window_label), 99)


def build_followup_timing_text(
    followup_window_label: str | None,
    monitoring_need_level: str | None = None,
    bundle_family: str | None = None,
) -> str:
    label = str(followup_window_label or "unknown").strip().lower()
    monitoring = str(monitoring_need_level or "").strip().lower()
    family = str(bundle_family or "").strip()

    generic_base = {
        "within_1_month": "Consider follow-up within about 1 month.",
        "within_3_months": "Consider follow-up within about 3 months.",
        "within_6_months": "Consider follow-up within about 6 months.",
        "routine_annual": "Routine annual follow-up is generally appropriate.",
        "unknown": "Follow-up timing should be interpreted using the broader clinical context.",
    }.get(label, "Follow-up timing should be interpreted using the broader clinical context.")

    family_templates = {
        "Monitoring": {
            "within_1_month": "Clinical review within about 1 month would be reasonable.",
            "within_3_months": "Clinical review within about 3 months would be reasonable.",
            "within_6_months": "Clinical review within about 6 months would be reasonable.",
            "routine_annual": "Routine annual clinical review is generally appropriate.",
        },
        "DietPattern": {
            "within_1_month": "Dietary follow-up within about 1 month may be helpful.",
            "within_3_months": "Dietary progress could be reviewed within about 3 months.",
            "within_6_months": "Dietary progress could be reviewed within about 6 months.",
            "routine_annual": "Routine annual dietary review is generally appropriate.",
        },
        "Movement": {
            "within_1_month": "Physical-activity follow-up within about 1 month may be helpful.",
            "within_3_months": "Physical-activity progress could be reviewed within about 3 months.",
            "within_6_months": "Physical-activity progress could be reviewed within about 6 months.",
            "routine_annual": "Routine annual activity review is generally appropriate.",
        },
        "Smoking": {
            "within_1_month": "Smoking-cessation follow-up within about 1 month may be helpful.",
            "within_3_months": "Smoking-cessation progress could be reviewed within about 3 months.",
            "within_6_months": "Smoking-related follow-up within about 6 months may be reasonable.",
            "routine_annual": "Routine annual smoking-status review is generally appropriate.",
        },
        "Weight": {
            "within_1_month": "Weight-management follow-up within about 1 month may be helpful.",
            "within_3_months": "Weight-related progress could be reviewed within about 3 months.",
            "within_6_months": "Weight-related progress could be reviewed within about 6 months.",
            "routine_annual": "Routine annual weight-related review is generally appropriate.",
        },
        "Lipids": {
            "within_1_month": "Lipid-related follow-up within about 1 month may be helpful.",
            "within_3_months": "Lipid-related progress could be reviewed within about 3 months.",
            "within_6_months": "Lipid-related progress could be reviewed within about 6 months.",
            "routine_annual": "Routine annual lipid review is generally appropriate.",
        },
        "Alcohol": {
            "within_1_month": "Alcohol-related follow-up within about 1 month may be helpful.",
            "within_3_months": "Alcohol-related progress could be reviewed within about 3 months.",
            "within_6_months": "Alcohol-related progress could be reviewed within about 6 months.",
            "routine_annual": "Routine annual alcohol-risk review is generally appropriate.",
        },
        "IntegratedCardiometabolic": {
            "within_1_month": "Broader cardiometabolic review within about 1 month may be appropriate.",
            "within_3_months": "Broader cardiometabolic review within about 3 months would be reasonable.",
            "within_6_months": "Broader cardiometabolic review within about 6 months would be reasonable.",
            "routine_annual": "Routine annual cardiometabolic review is generally appropriate.",
        },
    }

    family_base = family_templates.get(family, {}).get(label, generic_base)

    if monitoring in {"very_high", "high"} and label in {"within_1_month", "within_3_months"}:
        return f"{family_base} Earlier review may be appropriate if monitoring needs increase."

    return family_base


def build_followup_timing_note(
    followup_window_label: str | None,
    temporal_risk_tier: str | None = None,
    monitoring_need_level: str | None = None,
) -> str:
    label = str(followup_window_label or "unknown").strip().lower()
    risk = str(temporal_risk_tier or "").strip().lower()
    monitoring = str(monitoring_need_level or "").strip().lower()

    label_text = {
        "within_1_month": "short-term follow-up",
        "within_3_months": "near-term follow-up",
        "within_6_months": "medium-term follow-up",
        "routine_annual": "routine follow-up",
        "unknown": "context-dependent follow-up",
    }.get(label, "context-dependent follow-up")

    detail_parts = [label_text]

    if risk:
        detail_parts.append(f"temporal risk tier: {risk}")
    if monitoring:
        detail_parts.append(f"monitoring need: {monitoring}")

    return "Suggested timing profile: " + "; ".join(detail_parts) + "."
