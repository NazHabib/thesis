from .config import (
    PATIENT_CONFIDENCE_TEXT,
    CLINICIAN_CONFIDENCE_TEXT,
)


def build_confidence_messages(confidence_label: str) -> dict:
    label = str(confidence_label or "low").lower()
    return {
        "patient_confidence_explanation": PATIENT_CONFIDENCE_TEXT.get(
            label,
            PATIENT_CONFIDENCE_TEXT["low"],
        ),
        "clinician_confidence_explanation": CLINICIAN_CONFIDENCE_TEXT.get(
            label,
            CLINICIAN_CONFIDENCE_TEXT["low"],
        ),
    }
