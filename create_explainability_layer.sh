#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="apps/explainability_layer"

mkdir -p \
  "$APP_ROOT/output" \
  "$APP_ROOT/management/commands" \
  "$APP_ROOT/services/explainability" \
  "$APP_ROOT/tests"

touch \
  "$APP_ROOT/__init__.py" \
  "$APP_ROOT/admin.py" \
  "$APP_ROOT/apps.py" \
  "$APP_ROOT/models.py" \
  "$APP_ROOT/views.py" \
  "$APP_ROOT/urls.py" \
  "$APP_ROOT/output/README.md" \
  "$APP_ROOT/management/__init__.py" \
  "$APP_ROOT/management/commands/__init__.py" \
  "$APP_ROOT/management/commands/run_explainability.py" \
  "$APP_ROOT/services/__init__.py" \
  "$APP_ROOT/services/explainability/__init__.py" \
  "$APP_ROOT/services/explainability/config.py" \
  "$APP_ROOT/services/explainability/contracts.py" \
  "$APP_ROOT/services/explainability/templates.py" \
  "$APP_ROOT/services/explainability/traceability.py" \
  "$APP_ROOT/services/explainability/confidence.py" \
  "$APP_ROOT/services/explainability/explainers.py" \
  "$APP_ROOT/services/explainability/service.py" \
  "$APP_ROOT/tests/__init__.py" \
  "$APP_ROOT/tests/test_service.py"

echo "Explainability layer scaffold created under $APP_ROOT"
