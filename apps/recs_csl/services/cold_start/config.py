from apps.common.clinical_safety.severity import SEVERITY_TO_RANK


FEATURE_TIER_MINIMAL = "minimal"
FEATURE_TIER_CORE = "core"
FEATURE_TIER_FULL = "full"

CONF_LOW = "low"
CONF_MEDIUM = "medium"
CONF_HIGH = "high"

SEVERITY_RANK = {
    "informational": SEVERITY_TO_RANK["informational"],
    "low": SEVERITY_TO_RANK["low"],
    "medium": SEVERITY_TO_RANK["moderate"],
    "moderate": SEVERITY_TO_RANK["moderate"],
    "high": SEVERITY_TO_RANK["high"],
    "critical": SEVERITY_TO_RANK["critical"],
}

CONF_TO_INT = {
    "low": 1,
    "medium": 2,
    "high": 3,
}

INT_TO_CONF = {v: k for k, v in CONF_TO_INT.items()}

FAMILIES = [
    "DietPattern",
    "Movement",
    "Weight",
    "Smoking",
    "Alcohol",
    "Lipids",
    "Monitoring",
    "IntegratedCardiometabolic",
]

COLD_START_VERSION = "csl_1.1.0"
MIN_BP_READINGS_FOR_CORE = 2
GENERATED_ESCALATE_CODE = "CSL_ESCALATE_TRIAGE_NO_RULE"
ALLOW_ONE_SUPPORT_ITEM_IN_TRIAGE = True
