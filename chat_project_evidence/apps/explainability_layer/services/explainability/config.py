EXPLAINABILITY_VERSION = "1.0.0"

PATIENT_CONFIDENCE_TEXT = {
    "high": "The system has strong supporting evidence for this recommendation.",
    "medium": "The system has moderate supporting evidence for this recommendation.",
    "low": "This recommendation is more precautionary and should be interpreted conservatively.",
}

CLINICIAN_CONFIDENCE_TEXT = {
    "high": "High confidence based on upstream rule strength, retained evidence, and available participant features.",
    "medium": "Moderate confidence; recommendation is supported but subject to data limitations or weaker supporting evidence.",
    "low": "Low confidence; recommendation retained conservatively and should be interpreted with caution.",
}

TRIAGE_PATIENT_TEXT = (
    "Because urgent clinical risk indicators were detected, the system prioritised immediate safety-oriented guidance."
)

TRIAGE_CLINICIAN_TEXT = (
    "Triage mode active. Detailed lifestyle tailoring was intentionally deprioritised in favour of urgent safety escalation."
)

COLDSTART_PATIENT_TEXT = (
    "Because limited information was available, the recommendation was generated using a more conservative approach."
)

COLDSTART_CLINICIAN_TEXT = (
    "Cold-start or sparse-data constraints were detected; confidence ceilings and conservative gating were applied upstream."
)
