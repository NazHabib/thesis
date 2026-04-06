from .config import (
    TRIAGE_PATIENT_TEXT,
    TRIAGE_CLINICIAN_TEXT,
    COLDSTART_PATIENT_TEXT,
    COLDSTART_CLINICIAN_TEXT,
)


def build_patient_why_text(row) -> str:
    title = row.get("bundle_title", "This recommendation")
    rationale = row.get("rationale", "")
    confidence_label = str(row.get("confidence_label", "low")).lower()

    parts = [f"{title} was selected to support safer self-management."]

    if rationale:
        parts.append(str(rationale).strip())

    if confidence_label == "high":
        parts.append("This recommendation is supported by strong available evidence.")
    elif confidence_label == "medium":
        parts.append("This recommendation is supported by moderate available evidence.")
    else:
        parts.append("This recommendation is conservative and should be interpreted cautiously.")

    if bool(row.get("triage_mode", False)):
        parts.append(TRIAGE_PATIENT_TEXT)

    coldstart_note_codes = row.get("coldstart_note_codes", [])
    if isinstance(coldstart_note_codes, list) and len(coldstart_note_codes) > 0:
        parts.append(COLDSTART_PATIENT_TEXT)

    return " ".join(parts)


def build_clinician_why_text(row) -> str:
    bundle_family = row.get("bundle_family", "Unknown")
    action_codes = row.get("action_codes", [])
    source_rule_ids = row.get("source_rule_ids", [])
    confidence_label = row.get("confidence_label", "low")
    bundle_score = row.get("bundle_score", None)

    parts = [
        f"Bundle family: {bundle_family}.",
        f"Action count: {len(action_codes) if isinstance(action_codes, list) else 0}.",
        f"Confidence label: {confidence_label}.",
    ]

    if bundle_score is not None:
        parts.append(f"Bundle score: {bundle_score}.")

    if isinstance(source_rule_ids, list) and source_rule_ids:
        parts.append(f"Primary rule provenance: {', '.join(map(str, source_rule_ids))}.")

    if bool(row.get('triage_mode', False)):
        parts.append(TRIAGE_CLINICIAN_TEXT)

    coldstart_note_codes = row.get("coldstart_note_codes", [])
    if isinstance(coldstart_note_codes, list) and len(coldstart_note_codes) > 0:
        parts.append(COLDSTART_CLINICIAN_TEXT)

    return " ".join(parts)
