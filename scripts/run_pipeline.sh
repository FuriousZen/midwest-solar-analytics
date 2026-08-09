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

echo "==> 1/8 ingest public permit + utility records"
$PY -m msa.ingest.build_dataset
$PY -m scripts.report_horizon_coverage

for SEG in $SEGMENTS; do
  echo "==> 2/8 [$SEG] baselines (5-fold CV)"
  $PY -m msa.train.baseline --segment "$SEG"
  echo "==> 3/8 [$SEG] Optuna study ($TRIALS trials x 5-fold CV)"
  $PY -m msa.train.tune --segment "$SEG" --trials "$TRIALS"
  echo "==> 4/8 [$SEG] final model + MLflow registration"
  $PY -m msa.train.final --segment "$SEG"
  echo "==> 5/8 [$SEG] SHAP attribution dashboard"
  $PY -m msa.explain.shap_report --segment "$SEG"
done

echo "==> 6/8 export MLflow run summary + benchmark the API"
$PY -m scripts.export_mlflow_summary
for SEG in $SEGMENTS; do $PY -m msa.bench.latency --segment "$SEG" --requests 2000; done

echo "==> 7/8 quantify the absolute-time-index ablation"
$PY -m scripts.ablate_time_features

echo "==> 8/8 regenerate docs/RESULTS.md from the measured artifacts"
$PY -m scripts.make_results_tables

echo
echo "done. Every number in README.md / docs/RESULTS.md now comes from reports/."
