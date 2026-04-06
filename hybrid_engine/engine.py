from dataclasses import dataclass
from typing import List

@dataclass
class Rec:
    title: str
    category: str
    priority: int
    rationale: str
    explanation: str
    source: str
    confidence: float

def generate_recommendations(user_context: dict) -> List[Rec]:
    # TODO: replace with real hybrid engine (rules + model + fusion)
    return [
        Rec(
            title="Increase daily walking to 30 minutes",
            category="lifestyle",
            priority=2,
            rationale="Low recent activity pattern (stub).",
            explanation="Based on recent activity trend and general guidelines. (stub XAI)",
            source="stub",
            confidence=0.55,
        ),
        Rec(
            title="Measure blood pressure twice daily for 7 days",
            category="monitoring",
            priority=2,
            rationale="Support HTN monitoring routine (stub).",
            explanation="Recommended due to monitoring schedule best practice. (stub XAI)",
            source="stub",
            confidence=0.60,
        ),
    ]
