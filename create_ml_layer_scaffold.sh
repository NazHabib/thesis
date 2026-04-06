#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="apps/ml_layer"

mkdir -p \
  "$APP_ROOT/output" \
  "$APP_ROOT/management/commands" \
  "$APP_ROOT/services/ml_reranking" \
  "$APP_ROOT/training" \
  "$APP_ROOT/schemas" \
  "$APP_ROOT/artifacts/calibrators" \
  "$APP_ROOT/artifacts/manifests" \
  "$APP_ROOT/artifacts/models" \
  "$APP_ROOT/artifacts/preprocessors"

touch \
  "$APP_ROOT/__init__.py" \
  "$APP_ROOT/admin.py" \
  "$APP_ROOT/apps.py" \
  "$APP_ROOT/models.py" \
  "$APP_ROOT/tests.py" \
  "$APP_ROOT/views.py" \
  "$APP_ROOT/output/manifest.json" \
  "$APP_ROOT/output/reranked_recommendations.jsonl" \
  "$APP_ROOT/management/__init__.py" \
  "$APP_ROOT/management/commands/__init__.py" \
  "$APP_ROOT/management/commands/run_ml_reranking.py" \
  "$APP_ROOT/management/commands/train_ml_reranker.py" \
  "$APP_ROOT/management/commands/validate_ml_artifacts.py" \
  "$APP_ROOT/services/__init__.py" \
  "$APP_ROOT/services/ml_reranking/__init__.py" \
  "$APP_ROOT/services/ml_reranking/artifact_loader.py" \
  "$APP_ROOT/services/ml_reranking/config.py" \
  "$APP_ROOT/services/ml_reranking/contracts.py" \
  "$APP_ROOT/services/ml_reranking/fallback.py" \
  "$APP_ROOT/services/ml_reranking/feature_builder.py" \
  "$APP_ROOT/services/ml_reranking/lambda_policy.py" \
  "$APP_ROOT/services/ml_reranking/schema_validator.py" \
  "$APP_ROOT/services/ml_reranking/service.py" \
  "$APP_ROOT/training/__init__.py" \
  "$APP_ROOT/training/build_dataset.py" \
  "$APP_ROOT/training/calibrate.py" \
  "$APP_ROOT/training/evaluate.py" \
  "$APP_ROOT/training/preprocess.py" \
  "$APP_ROOT/training/splitters.py" \
  "$APP_ROOT/training/train_hgb.py" \
  "$APP_ROOT/training/train_lgbm.py" \
  "$APP_ROOT/training/train_logreg.py" \
  "$APP_ROOT/schemas/feature_schema_v1.json" \
  "$APP_ROOT/schemas/label_schema_v1.json" \
  "$APP_ROOT/schemas/manifest_schema_v1.json"

echo "Created ML layer scaffold at $APP_ROOT"
