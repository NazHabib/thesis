import json
from pathlib import Path


class MLArtifactLoader:
    """
    Minimal artifact/manifest loader.

    For now this is metadata-first and conservative:
    - loads manifest if present
    - checks declared artifact paths
    - does not yet deserialize model binaries
    """

    @staticmethod
    def load_manifest(manifest_path: Path) -> dict:
        if not manifest_path.exists():
            return {}

        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    @staticmethod
    def artifact_paths_exist(manifest: dict) -> bool:
        model_path = MLArtifactLoader._safe_path(manifest.get("model_path"))
        preprocessor_path = MLArtifactLoader._safe_path(manifest.get("preprocessor_path"))
        calibrator_path = MLArtifactLoader._safe_path(manifest.get("calibrator_path"))

        return all([
            model_path is not None and model_path.exists(),
            preprocessor_path is not None and preprocessor_path.exists(),
            calibrator_path is not None and calibrator_path.exists(),
        ])

    @staticmethod
    def build_metadata(manifest_path: Path) -> dict:
        manifest = MLArtifactLoader.load_manifest(manifest_path)
        ready = MLArtifactLoader.artifact_paths_exist(manifest)

        return {
            "manifest": manifest,
            "artifacts_ready": ready,
            "model_version": manifest.get("model_version", "unavailable"),
            "schema_version": manifest.get("feature_schema_version", "unavailable"),
            "calibration_version": manifest.get("calibration_version", "unavailable"),
            "max_ml_contrib": float(manifest.get("max_ml_contrib", 20.0)),
        }

    @staticmethod
    def _safe_path(value: str | None) -> Path | None:
        if not value:
            return None
        return Path(value)
