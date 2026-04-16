FEATURE_TIER_MINIMAL = "minimal"
FEATURE_TIER_CORE = "core"
FEATURE_TIER_FULL = "full"

CONF_LOW = "low"
CONF_MEDIUM = "medium"
CONF_HIGH = "high"

SEVERITY_RANK = {
    "informational": 0,
    "low": 1,
    "moderate": 2,
    "medium": 2,   
    "high": 3,
    "critical": 4,
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

COLD_START_VERSION = "csl_1.0.0"
MIN_BP_READINGS_FOR_CORE = 2
GENERATED_ESCALATE_CODE = "CSL_ESCALATE_TRIAGE_NO_RULE"
ALLOW_ONE_SUPPORT_ITEM_IN_TRIAGE = True
