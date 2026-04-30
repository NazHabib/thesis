from typing import Any

from .config import BUNDLE_CATEGORY_MAP, BUNDLE_TITLE_MAP, CONFIDENCE_TO_NUM, NUM_TO_CONFIDENCE

from apps.common.clinical_safety.scheduling import (
    build_followup_timing_text,
    derive_followup_window_label,
)


def unique_flatten(values: list[Any]) -> list[Any]:
    seen = set()
    out = []

    for item in values:
        if isinstance(item, list):
            for sub in item:
                key = repr(sub)
                if key not in seen:
                    seen.add(key)
                    out.append(sub)
        else:
            key = repr(item)
            if key not in seen:
                seen.add(key)
                out.append(item)

    return out


def merge_evidence_lists(evidence_series) -> list[dict]:
    seen = set()
    out = []

    for evidence_list in evidence_series:
        if not isinstance(evidence_list, list):
            continue
        for item in evidence_list:
            if not isinstance(item, dict):
                continue
            key = (
                item.get("feature"),
                item.get("operator"),
                str(item.get("threshold")),
                str(item.get("passed")),
            )
            if key not in seen:
                seen.add(key)
                out.append(item)

    return out


def conservative_confidence(conf_series) -> str:
    nums = [
        CONFIDENCE_TO_NUM.get(str(v).lower(), 1)
        for v in conf_series
        if v is not None
    ]
    if not nums:
        return "low"
    return NUM_TO_CONFIDENCE[min(nums)]


def build_bundle_title(bundle_family: str, action_type: str, action_codes: list[str], tags: list[str]) -> str:
    if action_type == "escalate":
        return "Urgent escalation and immediate follow-up"

    tags_set = {str(t).lower() for t in (tags or [])}
    action_codes_set = {str(a).upper() for a in (action_codes or [])}

    if bundle_family == "Monitoring" and "a1c" in tags_set:
        return "Glycaemic follow-up and review"

    if bundle_family == "Monitoring" and ("bp" in tags_set or "htn" in tags_set):
        return "Blood pressure follow-up and review"

    if bundle_family == "Monitoring" and any("DIABETES_REVIEW" in code for code in action_codes_set):
        return "Glycaemic follow-up and review"

    if bundle_family == "DietPattern" and "sodium" in tags_set:
        return "Reduce sodium and improve dietary pattern"

    if bundle_family == "DietPattern":
        return "Improve dietary pattern"

    if bundle_family == "Movement":
        return "Increase physical activity"

    if bundle_family == "Smoking":
        return "Smoking cessation support"

    if bundle_family == "Weight":
        return "Weight management support"

    if bundle_family == "Lipids":
        return "Lipid risk reduction support"

    if bundle_family == "Alcohol":
        return "Reduce alcohol-related risk"

    if bundle_family == "IntegratedCardiometabolic":
        return "Integrated cardiometabolic risk reduction plan"

    return BUNDLE_TITLE_MAP.get(bundle_family, "Personalised recommendation bundle")


def build_bundle_category(bundle_family: str) -> str:
    return BUNDLE_CATEGORY_MAP.get(bundle_family, "lifestyle")


def build_rationale(bundle_family: str, tags: list[str], triage_mode: bool) -> str:
    tags_set = {str(t).lower() for t in (tags or [])}

    if triage_mode:
        return (
            "Safety-first triage is active, so urgent escalation and clinically important "
            "monitoring actions were prioritised."
        )

    if bundle_family == "DietPattern" and "sodium" in tags_set:
        return (
            "This bundle brings together dietary actions aimed at lowering sodium intake "
            "and improving overall diet quality while reducing message overload."
        )

    if bundle_family == "DietPattern":
        return (
            "This bundle brings together dietary actions intended to improve overall diet quality "
            "and reduce cardiometabolic risk."
        )

    if bundle_family == "Movement":
        return (
            "This bundle brings together physical-activity recommendations intended to support "
            "gradual, sustainable increases in movement."
        )

    if bundle_family == "Monitoring":
        return (
            "This bundle brings together follow-up and monitoring actions intended to support "
            "safer ongoing self-management and clinical review."
        )

    if bundle_family == "Smoking":
        return (
            "This bundle brings together smoking-related actions intended to reduce "
            "cardiovascular and metabolic risk."
        )

    if bundle_family == "Alcohol":
        return (
            "This bundle brings together alcohol-related actions intended to reduce "
            "overall cardiometabolic risk."
        )

    if bundle_family == "Weight":
        return (
            "This bundle brings together weight-related actions intended to support "
            "overall cardiometabolic risk reduction."
        )

    if bundle_family == "Lipids":
        return (
            "This bundle brings together lipid-related actions intended to support "
            "cardiovascular risk reduction."
        )

    if bundle_family == "IntegratedCardiometabolic":
        return (
            "This bundle combines cross-domain actions intended to address overall "
            "cardiometabolic risk in a coordinated way."
        )
    
    if bundle_family == "DiabetesSelfManagement":
        return (
            "This bundle brings together diabetes self-management actions intended to support "
            "safer day-to-day glycaemic management and sustained self-care."
        )

    if bundle_family == "DataQuality":
        return (
            "This bundle brings together data-quality and measurement-completion actions intended "
            "to improve the reliability of subsequent clinical interpretation and follow-up."
        )

    return "This bundle combines related recommendations into a coherent action plan."


def build_explanation(bundle_family: str, action_codes: list[str], confidence_label: str, tags: list[str]) -> str:
    tags_set = {str(t).lower() for t in (tags or [])}
    n_actions = len(action_codes)

    if bundle_family == "DietPattern" and "sodium" in tags_set:
        focus = "diet quality and sodium intake"
    elif bundle_family == "DietPattern":
        focus = "diet quality"
    elif bundle_family == "Movement":
        focus = "physical activity"
    elif bundle_family == "Monitoring":
        focus = "follow-up and clinical monitoring"
    elif bundle_family == "Smoking":
        focus = "smoking cessation"
    elif bundle_family == "Weight":
        focus = "weight-related lifestyle change"
    elif bundle_family == "Lipids":
        focus = "lipid-related cardiovascular risk"
    elif bundle_family == "Alcohol":
        focus = "alcohol-related risk reduction"
    elif bundle_family == "DiabetesSelfManagement":
        focus = "diabetes self-management"
    elif bundle_family == "DataQuality":
        focus = "completion of key health data"
    elif bundle_family == "IntegratedCardiometabolic":
        focus = "overall cardiometabolic risk reduction"
    else:
        focus = "cardiometabolic self-management"

    action_word = "action" if n_actions == 1 else "actions"

    return (
        f"This bundle combines {n_actions} related {action_word} focused on {focus}. "
        f"Overall confidence is {confidence_label}."
    )
    
def build_bundle_followup_window_label(
    followup_horizon_days,
    followup_urgency: str | None = None,
) -> str:
    return derive_followup_window_label(
        followup_horizon_days=followup_horizon_days,
        followup_urgency=followup_urgency,
    )


def build_bundle_followup_timing_text(
    followup_window_label: str,
    monitoring_need_level: str | None = None,
    bundle_family: str | None = None,
) -> str:
    return build_followup_timing_text(
        followup_window_label=followup_window_label,
        monitoring_need_level=monitoring_need_level,
        bundle_family=bundle_family,
    )
