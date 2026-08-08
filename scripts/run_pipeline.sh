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

echo "==> 1/7 ingest public permit + utility records"
$PY -m msa.ingest.build_dataset
$PY -m scripts.report_horizon_coverage

for SEG in $SEGMENTS; do
  echo "==> 2/7 [$SEG] baselines (5-fold CV)"
  $PY -m msa.train.baseline --segment "$SEG"
  echo "==> 3/7 [$SEG] Optuna study ($TRIALS trials x 5-fold CV)"
  $PY -m msa.train.tune --segment "$SEG" --trials "$TRIALS"
  echo "==> 4/7 [$SEG] final model + MLflow registration"
  $PY -m msa.train.final --segment "$SEG"
  echo "==> 5/7 [$SEG] SHAP attribution dashboard"
  $PY -m msa.explain.shap_report --segment "$SEG"
done

echo "==> 6/7 export MLflow run summary + benchmark the API"
$PY -m scripts.export_mlflow_summary
$PY -m msa.bench.latency --segment permit_to_completion --requests 2000

echo "==> 7/7 regenerate docs/RESULTS.md from the measured artifacts"
$PY -m scripts.make_results_tables

echo
echo "done. Every number in README.md / docs/RESULTS.md now comes from reports/."
