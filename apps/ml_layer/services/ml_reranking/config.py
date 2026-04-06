from pathlib import Path
from django.conf import settings


ML_RERANKING_VERSION = "0.1.0"

APP_DIR = Path(settings.BASE_DIR) / "apps" / "ml_layer"
ARTIFACTS_DIR = APP_DIR / "artifacts"
SCHEMAS_DIR = APP_DIR / "schemas"
OUTPUT_DIR = APP_DIR / "output"

ARTIFACT_MODELS_DIR = ARTIFACTS_DIR / "models"
ARTIFACT_PREPROCESSORS_DIR = ARTIFACTS_DIR / "preprocessors"
ARTIFACT_CALIBRATORS_DIR = ARTIFACTS_DIR / "calibrators"
ARTIFACT_MANIFESTS_DIR = ARTIFACTS_DIR / "manifests"

ACTIVE_MANIFEST_PATH = ARTIFACT_MANIFESTS_DIR / "active_manifest.json"

# Fixed system policy
LAMBDA_BY_FEATURE_TIER = {
    "minimal": 0.15,  # cold-start equivalent
    "core": 0.30,
    "full": 0.40,
}

LAMBDA_MIN = 0.0
LAMBDA_MAX = 0.50

# Conservative reduction rules
LOW_COMPLETENESS_THRESHOLD = 0.50
VERY_LOW_COMPLETENESS_THRESHOLD = 0.30
LOW_CONFIDENCE_THRESHOLD = 0.50

LAMBDA_PENALTY_LOW_COMPLETENESS = 0.05
LAMBDA_PENALTY_LOW_RULE_CONFIDENCE = 0.05

# ML contribution clipping: ML must not dominate rules
DEFAULT_MAX_ML_CONTRIB = 20.0

# Fallback settings
MIN_REQUIRED_CANDIDATES_FOR_ML = 2
LOW_SCORE_SPREAD_THRESHOLD = 0.03

# Artifact metadata defaults
DEFAULT_MODEL_VERSION = "unavailable"
DEFAULT_SCHEMA_VERSION = "unavailable"
DEFAULT_CALIBRATION_VERSION = "unavailable"
