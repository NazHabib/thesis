import pandas as pd

from apps.bundling_layer.services.bundling.service import run_bundling_layer


def test_bundling_layer_runs():
    safety_context_df = pd.DataFrame([
        {
            "ParticipantKey": "P1",
            "triage_mode": False,
            "triage_reason": "none",
            "feature_tier": "full",
            "confidence_ceiling_global": "high",
        }
    ])

    ranked_df = pd.DataFrame([
        {
            "ParticipantKey": "P1",
            "action_code": "RECOMMEND_DASH_OR_MEDITERRANEAN_PATTERN",
            "action_type": "recommend",
            "severity": "moderate",
            "priority": 580,
            "tags": ["diet", "pattern"],
            "evidence": [],
            "source_rule_ids": ["R1"],
            "bundle_family": "DietPattern",
            "coldstart_note_codes": [],
            "coldstart_confidence_ceiling": "high",
            "total_score": 150.0,
            "confidence_score": 0.9,
            "confidence_label": "high",
        },
        {
            "ParticipantKey": "P1",
            "action_code": "INFORM_VERY_HIGH_SODIUM_DAY1",
            "action_type": "inform",
            "severity": "high",
            "priority": 680,
            "tags": ["diet", "sodium"],
            "evidence": [],
            "source_rule_ids": ["R2"],
            "bundle_family": "DietPattern",
            "coldstart_note_codes": [],
            "coldstart_confidence_ceiling": "high",
            "total_score": 140.0,
            "confidence_score": 0.8,
            "confidence_label": "medium",
        },
    ])

    result = run_bundling_layer(safety_context_df, ranked_df)

    assert "final_recommendations" in result
    assert "manifest" in result
    assert len(result["final_recommendations"]) == 1
    assert result["final_recommendations"].iloc[0]["bundle_family"] == "DietPattern"
