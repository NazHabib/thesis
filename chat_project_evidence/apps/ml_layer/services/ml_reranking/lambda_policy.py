from .config import (
    LAMBDA_BY_FEATURE_TIER,
    LAMBDA_MAX,
    LAMBDA_MIN,
    LOW_COMPLETENESS_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
    LAMBDA_PENALTY_LOW_COMPLETENESS,
    LAMBDA_PENALTY_LOW_RULE_CONFIDENCE,
)


def compute_lambda(
    triage_mode: bool,
    feature_tier: str,
    data_completeness_score: float | int | None,
    max_rule_confidence_score: float | int | None,
) -> float:
    """
    Dynamic ML influence policy.

    Policy:
    - triage disables ML influence
    - base lambda depends on feature tier
    - penalties reduce lambda under lower completeness or lower rule confidence
    - final value is clipped to the configured safety bounds
    """
    if bool(triage_mode):
        return 0.0

    tier = str(feature_tier).strip().lower() if feature_tier is not None else "minimal"
    base = float(LAMBDA_BY_FEATURE_TIER.get(tier, LAMBDA_BY_FEATURE_TIER["minimal"]))

    completeness = _safe_float(data_completeness_score, default=0.0)
    rule_conf = _safe_float(max_rule_confidence_score, default=0.0)

    if completeness < LOW_COMPLETENESS_THRESHOLD:
        base -= LAMBDA_PENALTY_LOW_COMPLETENESS

    if rule_conf < LOW_CONFIDENCE_THRESHOLD:
        base -= LAMBDA_PENALTY_LOW_RULE_CONFIDENCE

    return float(max(LAMBDA_MIN, min(base, LAMBDA_MAX)))


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
