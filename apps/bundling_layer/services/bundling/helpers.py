from typing import Any

from .config import BUNDLE_CATEGORY_MAP, BUNDLE_TITLE_MAP, CONFIDENCE_TO_NUM, NUM_TO_CONFIDENCE


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

    if bundle_family == "Monitoring" and "a1c" in tags_set:
        return "Glycaemic follow-up and review"
    if bundle_family == "Monitoring" and ("bp" in tags_set or "htn" in tags_set):
        return "Blood pressure follow-up and review"
    if bundle_family == "DietPattern" and "sodium" in tags_set:
        return "Reduce sodium and improve dietary pattern"
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

    return BUNDLE_TITLE_MAP.get(bundle_family, "Personalised recommendation bundle")


def build_bundle_category(bundle_family: str) -> str:
    return BUNDLE_CATEGORY_MAP.get(bundle_family, "lifestyle")


def build_rationale(bundle_family: str, tags: list[str], triage_mode: bool) -> str:
    if triage_mode:
        return "Safety-first triage is active; urgent and monitoring-oriented recommendations were prioritised."

    if bundle_family == "DietPattern":
        return "This bundle consolidates dietary recommendations to reduce cardiometabolic risk and avoid message overload."
    if bundle_family == "Movement":
        return "This bundle consolidates movement recommendations to support gradual and sustainable activity changes."
    if bundle_family == "Monitoring":
        return "This bundle consolidates follow-up and monitoring actions to support safer self-management."
    if bundle_family == "Smoking":
        return "This bundle consolidates smoking-related advice to reduce cardiovascular and metabolic risk."
    if bundle_family == "Alcohol":
        return "This bundle consolidates alcohol-related advice to reduce cardiometabolic risk."
    if bundle_family == "Weight":
        return "This bundle consolidates weight-related recommendations to support overall risk reduction."
    if bundle_family == "Lipids":
        return "This bundle consolidates lipid-related recommendations to support cardiovascular risk reduction."

    return "This bundle consolidates related recommendations into a coherent action plan."


def build_explanation(bundle_family: str, action_codes: list[str], confidence_label: str, tags: list[str]) -> str:
    tags_set = {str(t).lower() for t in (tags or [])}
    n_actions = len(action_codes)

    if bundle_family == "DietPattern" and "sodium" in tags_set:
        focus = "diet quality and sodium intake"
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
    else:
        focus = "cardiometabolic self-management"

    return (
        f"This bundle merges {n_actions} related recommendation(s) focused on {focus}. "
        f"Final confidence is {confidence_label}."
    )
