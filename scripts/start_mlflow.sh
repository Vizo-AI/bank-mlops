#!/usr/bin/env bash
set -euo pipefail

DB_PATH="mlflow.db"
ARTIFACT_ROOT="mlruns"
PORT="${MLFLOW_PORT:-5000}"

mkdir -p "$ARTIFACT_ROOT"

mlflow ui \
  --backend-store-uri "sqlite:///$DB_PATH" \
  --default-artifact-root "$ARTIFACT_ROOT" \
  --host 0.0.0.0 \
  --port "$PORT"