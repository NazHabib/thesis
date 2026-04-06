import pandas as pd

from apps.explainability_layer.services.explainability.service import run_explainability_layer


def test_run_explainability_layer_basic():
    safety_context_df = pd.DataFrame([
        {
            "ParticipantKey": "P1",
            "feature_tier": "full",
            "triage_mode": False,
            "triage_reason": "none",
            "confidence_ceiling_global": "high",
        }
    ])

    final_recommendations_df = pd.DataFrame([
        {
            "ParticipantKey": "P1",
            "bundle_title": "Increase physical activity",
            "bundle_family": "Movement",
            "category": "lifestyle",
            "action_codes": ["RECOMMEND_MVPA_START_GRADUAL"],
            "action_type": "recommend",
            "priority": 520,
            "tags": ["physical_activity", "mvpa"],
            "source_rule_ids": ["PA-0701"],
            "evidence": [],
            "coldstart_note_codes": [],
            "bundle_score": 136.0,
            "confidence_score": 0.82,
            "confidence_label": "high",
            "rationale": "Movement support is appropriate.",
            "explanation": "Bundle explanation.",
            "triage_mode": False,
            "triage_reason": "none",
        }
    ])

    out = run_explainability_layer(
        safety_context_df=safety_context_df,
        final_recommendations_df=final_recommendations_df,
    )

    explained_df = out["explained_recommendations"]

    assert len(explained_df) == 1
    assert "patient_facing_explanation" in explained_df.columns
    assert "clinician_facing_explanation" in explained_df.columns
    assert "traceability_payload" in explained_df.columns
