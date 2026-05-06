import pandas as pd

from apps.explainability_layer.services.explainability.service import run_explainability_layer

from apps.explainability_layer.services.explainability.service import (
    run_explainability_layer,
    build_patient_safe_explainability_view,
    build_clinician_explainability_view,
)

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
    
def test_triage_patient_explanation_is_safety_first():
    safety_context_df = pd.DataFrame([{
        "ParticipantKey": "P1",
        "feature_tier": "core",
        "triage_mode": True,
        "triage_reason": "hypertensive_crisis",
        "confidence_ceiling_global": "high",
    }])

    final_recommendations_df = pd.DataFrame([{
        "ParticipantKey": "P1",
        "bundle_title": "Urgent escalation and immediate follow-up",
        "bundle_family": "Monitoring",
        "category": "monitoring",
        "action_codes": ["ESCALATE_HTN_CRISIS"],
        "action_type": "escalate",
        "priority": 1050,
        "tags": ["acute", "safety", "htn"],
        "source_rule_ids": ["HTNACUTE-0202"],
        "evidence": [],
        "coldstart_note_codes": [],
        "bundle_score": 200.0,
        "confidence_score": 1.0,
        "confidence_label": "high",
        "rationale": "Safety-first triage is active.",
        "explanation": "Urgent escalation bundle.",
        "triage_mode": True,
        "triage_reason": "hypertensive_crisis",
    }])

    out = run_explainability_layer(safety_context_df, final_recommendations_df)
    row = out["explained_recommendations"].iloc[0]

    assert "safety-first" in row["patient_facing_explanation"].lower()
    assert "urgent" in row["patient_facing_explanation"].lower() or "higher-risk" in row["patient_facing_explanation"].lower()
    assert row["traceability_payload"]["triage_mode"] is True
    
def test_explainability_without_xai_inputs_uses_safe_fallback():
    safety_context_df = pd.DataFrame([{
        "ParticipantKey": "P1",
        "feature_tier": "minimal",
        "triage_mode": False,
        "triage_reason": "none",
        "confidence_ceiling_global": "medium",
    }])

    final_recommendations_df = pd.DataFrame([{
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
        "coldstart_note_codes": ["CS_NOTE_COLD_START_MINIMAL"],
        "bundle_score": 100.0,
        "confidence_score": 0.5,
        "confidence_label": "medium",
        "rationale": "Movement support is appropriate.",
        "explanation": "Bundle explanation.",
        "triage_mode": False,
        "triage_reason": "none",
    }])

    out = run_explainability_layer(safety_context_df, final_recommendations_df)
    row = out["explained_recommendations"].iloc[0]

    assert isinstance(row["ml_xai_payload"], dict)
    assert row["ml_xai_payload"]["xai_status"] in {"missing_inputs", "shap_not_installed"} or "xai" in row["ml_xai_payload"]["xai_status"]
    assert "detailed model-based explanation was not available" in row["patient_ml_xai_explanation"].lower()
    
def test_patient_text_does_not_expose_rule_ids_or_shap_values():
    safety_context_df = pd.DataFrame([{
        "ParticipantKey": "P1",
        "feature_tier": "full",
        "triage_mode": False,
        "triage_reason": "none",
        "confidence_ceiling_global": "high",
    }])

    final_recommendations_df = pd.DataFrame([{
        "ParticipantKey": "P1",
        "bundle_title": "Reduce sodium and improve dietary pattern",
        "bundle_family": "DietPattern",
        "category": "lifestyle",
        "action_codes": ["RECOMMEND_SODIUM_REDUCTION_WHO_2000_DAY1"],
        "action_type": "recommend",
        "priority": 520,
        "tags": ["diet", "sodium"],
        "source_rule_ids": ["DIET-0602"],
        "evidence": [],
        "coldstart_note_codes": [],
        "bundle_score": 120.0,
        "confidence_score": 0.75,
        "confidence_label": "medium",
        "rationale": "This bundle brings together dietary actions.",
        "explanation": "Bundle explanation.",
        "triage_mode": False,
        "triage_reason": "none",
    }])

    out = run_explainability_layer(safety_context_df, final_recommendations_df)
    row = out["explained_recommendations"].iloc[0]

    patient_text = " ".join([
        str(row["patient_facing_explanation"]),
        str(row["patient_confidence_explanation"]),
        str(row["patient_ml_xai_explanation"]),
    ])

    assert "DIET-0602" not in patient_text
    assert "shap" not in patient_text.lower()
    assert "source_rule" not in patient_text.lower()
    
def test_patient_triage_reason_is_human_readable():
    safety_context_df = pd.DataFrame([
        {
            "ParticipantKey": "P1",
            "feature_tier": "core",
            "triage_mode": True,
            "triage_reason": "hypertensive_crisis",
            "confidence_ceiling_global": "high",
        }
    ])

    final_recommendations_df = pd.DataFrame([
        {
            "ParticipantKey": "P1",
            "bundle_title": "Urgent escalation and immediate follow-up",
            "bundle_family": "Monitoring",
            "category": "monitoring",
            "action_codes": ["ESCALATE_HTN_CRISIS"],
            "action_type": "escalate",
            "priority": 1050,
            "tags": ["acute", "safety", "htn"],
            "source_rule_ids": ["HTNACUTE-0202"],
            "evidence": [],
            "coldstart_note_codes": [],
            "bundle_score": 200.0,
            "confidence_score": 1.0,
            "confidence_label": "high",
            "rationale": "Safety-first triage is active.",
            "explanation": "Urgent escalation bundle.",
            "triage_mode": True,
            "triage_reason": "hypertensive_crisis",
        }
    ])

    out = run_explainability_layer(
        safety_context_df=safety_context_df,
        final_recommendations_df=final_recommendations_df,
    )

    text = out["explained_recommendations"].iloc[0]["patient_facing_explanation"]

    assert "very high blood pressure indicators" in text
    assert "hypertensive_crisis" not in text
