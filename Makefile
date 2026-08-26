# Reproducible entrypoints for every stage of the pipeline.
PY ?= .venv/bin/python
TRIALS ?= 200
SEGMENT ?= permit_to_completion

.PHONY: help venv dev data baselines tune final explain serve bench test lint \
        audit smoke ablate mlflow-summary pipeline mlflow-ui clean

help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | column -t -s "$$(printf '\t')"

venv: ## create .venv and install pinned dependencies
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.lock.txt
	.venv/bin/pip install --no-deps -e .

dev: ## add the linter / auditor on top of .venv
	.venv/bin/pip install -r requirements-dev.txt

data: ## download and assemble the public permit + utility corpus
	$(PY) -m msa.ingest.build_dataset
	$(PY) -m scripts.report_horizon_coverage

baselines: ## 5-fold CV for all four baselines
	$(PY) -m msa.train.baseline --segment $(SEGMENT)

tune: ## Optuna study (override with TRIALS=n)
	$(PY) -m msa.train.tune --segment $(SEGMENT) --trials $(TRIALS)

final: ## fit the tuned model, score the holdout, register with MLflow
	$(PY) -m msa.train.final --segment $(SEGMENT)

explain: ## build the SHAP attribution dashboard
	$(PY) -m msa.explain.shap_report --segment $(SEGMENT)

serve: ## run the prediction API on :8000
	.venv/bin/uvicorn msa.serve.app:app --host 127.0.0.1 --port 8000

bench: ## measure the latency distribution
	$(PY) -m msa.bench.latency --segment $(SEGMENT) --requests 2000

test: ## run the test suite
	$(PY) -m pytest

lint: ## static checks (same command CI runs)
	.venv/bin/ruff check .

audit: ## known CVEs in the pinned environment (see docs/SECURITY.md)
	.venv/bin/pip-audit -r requirements.lock.txt --progress-spinner off

smoke: ## whole pipeline end to end, offline, in a temp dir (~3 min)
	$(PY) -m scripts.smoke_pipeline --trials 3

ablate: ## compare the model with and without absolute time indices
	$(PY) -m scripts.ablate_time_features

mlflow-summary: ## export the MLflow run summary to reports/
	$(PY) -m scripts.export_mlflow_summary

pipeline: ## everything, both segments, end to end
	./scripts/run_pipeline.sh

mlflow-ui: ## browse the tracked runs
	.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db

clean: ## remove generated models/reports (keeps cached raw downloads)
	rm -rf models reports/permit_to_completion reports/application_to_issue
