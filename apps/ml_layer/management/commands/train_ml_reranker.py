import json
from pathlib import Path

import joblib
import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand

from apps.ml_layer.training.build_dataset import build_ml_training_dataset
from apps.ml_layer.training.splitters import make_stratified_participant_level_split
from apps.ml_layer.training.preprocess import prepare_lgbm_features
from apps.ml_layer.training.train_lgbm import train_lgbm_binary_classifier
from apps.ml_layer.training.calibrate import calibrate_probabilities
from apps.ml_layer.training.feature_groups import (
    temporal_feature_count,
    temporal_feature_subset,
)

class Command(BaseCommand):
    help = "Train, calibrate, and persist the ML reranker artifacts."

    def add_arguments(self, parser):
        parser.add_argument(
            "--safety-context",
            type=str,
            default="apps/recs_csl/output/safety_context.jsonl",
        )
        parser.add_argument(
            "--feature-profile",
            type=str,
            default="default",
            choices=["default", "experiment_c", "temporal_full", "no_temporal", "temporal_only"],
        )
        parser.add_argument(
            "--ranked",
            type=str,
            default="apps/ranking_layer/output/ranked_recommendations.jsonl",
        )
        parser.add_argument(
            "--participant-features",
            type=str,
            default="nhanes_step1_out/NHANES_Master_engineered.parquet",
        )
        parser.add_argument(
            "--top-k-positive",
            type=int,
            default=3,
        )
        parser.add_argument(
            "--calibration-method",
            type=str,
            default="isotonic",
            choices=["isotonic", "platt"],
        )
        parser.add_argument(
            "--model-version",
            type=str,
            default="v1.0.0",
        )
        parser.add_argument(
            "--schema-version",
            type=str,
            default="v1",
        )
        parser.add_argument(
            "--calibration-version",
            type=str,
            default="v1",
        )
        parser.add_argument(
            "--max-ml-contrib",
            type=float,
            default=0.5,
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
        )
        parser.add_argument(
            "--label-scheme",
            type=str,
            default="rule_top_k",
            choices=[
                "rule_top_k",
                "rule_top_k_with_min_score",
                "rule_top_k_plus_bundle_retention",
            ],
        )
        parser.add_argument(
            "--min-score-norm",
            type=float,
            default=0.80,
        )

    def handle(self, *args, **options):
        base_dir = Path(settings.BASE_DIR)

        safety_path = base_dir / options["safety_context"]
        ranked_path = base_dir / options["ranked"]
        participant_features_path = base_dir / options["participant_features"]

        self.stdout.write("Loading training inputs...")
        safety_df = pd.read_json(safety_path, lines=True)
        ranked_df = pd.read_json(ranked_path, lines=True)
        participant_features_df = pd.read_parquet(participant_features_path)

        self.stdout.write("Building ML training dataset...")
        train_df = build_ml_training_dataset(
            safety_context_df=safety_df,
            ranked_df=ranked_df,
            participant_features_df=participant_features_df,
            top_k_positive=int(options["top_k_positive"]),
            exclude_triage=True,
            label_scheme=options["label_scheme"],
            min_score_norm=float(options["min_score_norm"]),
        )
        self.stdout.write("Creating participant-level split...")
        split = make_stratified_participant_level_split(
            df=train_df,
            stratify_col="feature_tier",
            participant_col="ParticipantKey",
            seed=int(options["seed"]),
        )

        self.stdout.write("Preparing LightGBM features...")
        prepared = prepare_lgbm_features(
            train_df=split.train_df,
            val_df=split.val_df,
            target_col="label",
            feature_profile=options["feature_profile"],
        )

        self.stdout.write("Training LightGBM model...")
        train_result = train_lgbm_binary_classifier(
            X_train=prepared.X_train,
            y_train=prepared.y_train,
            X_val=prepared.X_val,
            y_val=prepared.y_val,
            random_state=int(options["seed"]),
        )

        self.stdout.write("Calibrating validation probabilities...")
        calibration_result = calibrate_probabilities(
            y_val=prepared.y_val,
            val_pred_raw=train_result.val_pred_proba,
            method=options["calibration_method"],
        )

        artifacts_dir = base_dir / "apps" / "ml_layer" / "artifacts"
        models_dir = artifacts_dir / "models"
        preprocessors_dir = artifacts_dir / "preprocessors"
        calibrators_dir = artifacts_dir / "calibrators"
        manifests_dir = artifacts_dir / "manifests"

        models_dir.mkdir(parents=True, exist_ok=True)
        preprocessors_dir.mkdir(parents=True, exist_ok=True)
        calibrators_dir.mkdir(parents=True, exist_ok=True)
        manifests_dir.mkdir(parents=True, exist_ok=True)

        model_version = options["model_version"]
        schema_version = options["schema_version"]
        calibration_version = options["calibration_version"]

        model_path = models_dir / f"lgbm_model_{model_version}.joblib"
        preprocessor_path = preprocessors_dir / f"lgbm_preprocessor_{model_version}.joblib"
        calibrator_path = calibrators_dir / f"calibrator_{options['calibration_method']}_{calibration_version}.joblib"

        self.stdout.write("Saving artifacts...")
        joblib.dump(train_result.model, model_path)
        joblib.dump(prepared.preprocessor, preprocessor_path)
        joblib.dump(calibration_result.calibrator, calibrator_path)

        manifest = {
            "model_name": "lightgbm_reranker",
            "model_version": model_version,
            "feature_schema_version": schema_version,
            "calibration_version": calibration_version,
            "calibration_method": calibration_result.method,
            "model_path": str(model_path),
            "preprocessor_path": str(preprocessor_path),
            "calibrator_path": str(calibrator_path),
            "max_ml_contrib": float(options["max_ml_contrib"]),
            "training": {
            "feature_profile": options["feature_profile"],
            "top_k_positive": int(options["top_k_positive"]),
            "seed": int(options["seed"]),
            "train_rows": int(len(split.train_df)),
            "val_rows": int(len(split.val_df)),
            "test_rows": int(len(split.test_df)),
            "train_participants": int(len(split.train_keys)),
            "val_participants": int(len(split.val_keys)),
            "test_participants": int(len(split.test_keys)),
            "feature_columns": prepared.feature_columns,
            "categorical_columns": prepared.categorical_columns,
            "numeric_columns": prepared.numeric_columns,
            "temporal_feature_columns": temporal_feature_subset(prepared.feature_columns),
            "temporal_feature_count": int(temporal_feature_count(prepared.feature_columns)),
            "label_scheme": options["label_scheme"],
            "min_score_norm": float(options["min_score_norm"]),
        },
            "metrics": {
                **train_result.metrics,
                **calibration_result.metrics,
            },
        }

        versioned_manifest_path = manifests_dir / f"manifest_{model_version}.json"
        active_manifest_path = manifests_dir / "active_manifest.json"

        with open(versioned_manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        with open(active_manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        self.stdout.write(self.style.SUCCESS("ML reranker training complete."))
        self.stdout.write(f"Saved model: {model_path}")
        self.stdout.write(f"Saved preprocessor: {preprocessor_path}")
        self.stdout.write(f"Saved calibrator: {calibrator_path}")
        self.stdout.write(f"Saved active manifest: {active_manifest_path}")
        self.stdout.write("Training metrics:")
        for k, v in manifest["metrics"].items():
            self.stdout.write(f"  - {k}: {v}")
