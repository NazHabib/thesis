BUNDLING_VERSION = "bundling_1.0.0"

MAX_FINAL_BUNDLES_PER_PARTICIPANT = 5

CONFIDENCE_TO_NUM = {
    "low": 1,
    "medium": 2,
    "high": 3,
}

NUM_TO_CONFIDENCE = {v: k for k, v in CONFIDENCE_TO_NUM.items()}

BUNDLE_TITLE_MAP = {
    "DietPattern": "Improve dietary pattern",
    "Movement": "Increase physical activity",
    "Weight": "Weight management support",
    "Smoking": "Smoking cessation support",
    "Alcohol": "Reduce alcohol-related risk",
    "Lipids": "Lipid and cardiometabolic risk support",
    "Monitoring": "Monitoring and follow-up",
    "DiabetesSelfManagement": "Diabetes self-management support",
    "DataQuality": "Complete key health data",
    "IntegratedCardiometabolic": "Integrated cardiometabolic support",
}

BUNDLE_CATEGORY_MAP = {
    "DietPattern": "lifestyle",
    "Movement": "lifestyle",
    "Weight": "lifestyle",
    "Smoking": "lifestyle",
    "Alcohol": "lifestyle",
    "Lipids": "lifestyle",
    "Monitoring": "monitoring",
    "DiabetesSelfManagement": "self_management",
    "DataQuality": "monitoring",
    "IntegratedCardiometabolic": "lifestyle",
}
