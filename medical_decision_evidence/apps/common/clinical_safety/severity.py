from __future__ import annotations


CANONICAL_SEVERITY_ORDER = [
    "informational",
    "low",
    "moderate",
    "high",
    "critical",
]

CANONICAL_SEVERITY_SET = set(CANONICAL_SEVERITY_ORDER)

SEVERITY_TO_RANK = {
    "informational": 0,
    "low": 1,
    "moderate": 2,
    "high": 3,
    "critical": 4,
}

SEVERITY_TO_SCORE = {
    "informational": 0,
    "low": 10,
    "moderate": 40,
    "high": 70,
    "critical": 100,
}

SEVERITY_ALIASES = {
    "info": "informational",
    "informational": "informational",
    "low": "low",
    "moderate": "moderate",
    "medium": "moderate",
    "high": "high",
    "critical": "critical",
}


def normalize_severity(value: str | None, default: str = "low") -> str:
    if value is None:
        return default

    key = str(value).strip().lower()
    if key in SEVERITY_ALIASES:
        return SEVERITY_ALIASES[key]

    return default


def is_valid_severity(value: str | None) -> bool:
    if value is None:
        return False

    key = str(value).strip().lower()
    return key in SEVERITY_ALIASES


def severity_rank(value: str | None, default: str = "low") -> int:
    sev = normalize_severity(value, default=default)
    return SEVERITY_TO_RANK[sev]


def severity_score(value: str | None, default: str = "low") -> int:
    sev = normalize_severity(value, default=default)
    return SEVERITY_TO_SCORE[sev]
