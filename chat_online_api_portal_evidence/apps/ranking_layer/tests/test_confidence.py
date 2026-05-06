import pandas as pd

from apps.ranking_layer.services.ranking.confidence import apply_confidence


def test_confidence_is_capped_by_coldstart_ceiling():
    df = pd.DataFrame([
        {
            "ParticipantKey": "P1",
            "action_code": "A_HIGH",
            "total_score": 100.0,
            "coldstart_confidence_ceiling": "low",
        },
        {
            "ParticipantKey": "P1",
            "action_code": "A_LOW",
            "total_score": 10.0,
            "coldstart_confidence_ceiling": "high",
        },
    ])

    out = apply_confidence(df)

    high_row = out.loc[out["action_code"] == "A_HIGH"].iloc[0]
    assert high_row["confidence_label_raw"] == "high"
    assert high_row["confidence_label"] == "low"


def test_single_candidate_gets_mid_normalized_score():
    df = pd.DataFrame([
        {
            "ParticipantKey": "P1",
            "action_code": "ONLY",
            "total_score": 42.0,
            "coldstart_confidence_ceiling": "high",
        }
    ])

    out = apply_confidence(df)
    assert out.loc[0, "score_norm"] == 0.5
    assert out.loc[0, "confidence_label"] == "medium"
