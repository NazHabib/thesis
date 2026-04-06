SEVERITY_SCORE_MAP = {
    "critical": 100,
    "high": 70,
    "moderate": 40,
    "low": 10,
}

CONFIDENCE_TO_NUM = {
    "low": 1,
    "medium": 2,
    "high": 3,
}

NUM_TO_CONFIDENCE = {v: k for k, v in CONFIDENCE_TO_NUM.items()}

RANKING_VERSION = "ranking_1.0.0"

MAX_RECOMMENDATIONS_PER_PARTICIPANT = 5
EVIDENCE_BONUS_PER_PASSED = 10
CONTRAINDICATION_UNKNOWN_PENALTY = 20
MISSING_DATA_PENALTY = 10
