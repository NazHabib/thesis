def build_traceability_payload(row) -> dict:
    return {
        "bundle_family": row.get("bundle_family"),
        "action_codes": row.get("action_codes", []),
        "source_rule_ids": row.get("source_rule_ids", []),
        "confidence_label": row.get("confidence_label"),
        "confidence_score": row.get("confidence_score"),
        "bundle_score": row.get("bundle_score"),
        "triage_mode": bool(row.get("triage_mode", False)),
        "triage_reason": row.get("triage_reason", "none"),
        "coldstart_note_codes": row.get("coldstart_note_codes", []),
    }
