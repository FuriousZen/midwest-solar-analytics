#!/usr/bin/env bash
# Reproduce the entire project end to end.
#
#   ./scripts/run_pipeline.sh            # both segments, 200 Optuna trials each
#   TRIALS=20 ./scripts/run_pipeline.sh  # quick smoke run
#
# Requires the virtualenv at .venv (see README).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
TRIALS=${TRIALS:-200}
SEGMENTS=${SEGMENTS:-"permit_to_completion application_to_issue"}

echo "==> 1/6 ingest public permit + utility records"
$PY -m msa.ingest.build_dataset

for SEG in $SEGMENTS; do
  echo "==> 2/6 [$SEG] baselines (5-fold CV)"
  $PY -m msa.train.baseline --segment "$SEG"
  echo "==> 3/6 [$SEG] Optuna study ($TRIALS trials x 5-fold CV)"
  $PY -m msa.train.tune --segment "$SEG" --trials "$TRIALS"
  echo "==> 4/6 [$SEG] final model + MLflow registration"
  $PY -m msa.train.final --segment "$SEG"
  echo "==> 5/6 [$SEG] SHAP attribution dashboard"
  $PY -m msa.explain.shap_report --segment "$SEG"
done

echo "==> 6/6 export MLflow run summary + benchmark the API"
$PY -m scripts.export_mlflow_summary
$PY -m msa.bench.latency

echo "done. see reports/ for every number quoted in the README."
