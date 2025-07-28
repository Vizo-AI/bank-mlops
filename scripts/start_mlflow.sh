#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
  echo "Usage: start_mlflow.sh [--help]"
  echo "Starts MLflow UI with default settings."
  exit 0
fi

DB_PATH="mlflow.db"
ARTIFACT_ROOT="mlruns"
PORT="${MLFLOW_PORT:-8080}"
mkdir -p "$ARTIFACT_ROOT"

mlflow ui \
  --backend-store-uri "sqlite:///$DB_PATH" \
  --default-artifact-root "$ARTIFACT_ROOT" \
  --host 0.0.0.0 \
  --port "$PORT"
