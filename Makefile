# Reproducible entrypoints. Every number in README.md comes from one of these.
PY ?= .venv/bin/python
TRIALS ?= 200
SEGMENT ?= permit_to_completion

.PHONY: help venv data baselines tune final explain serve bench test pipeline clean

help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | column -t -s "$$(printf '\t')"

venv: ## create .venv and install pinned dependencies
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install -e .

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
	$(PY) -m pytest -q

pipeline: ## everything, both segments, end to end
	./scripts/run_pipeline.sh

mlflow-ui: ## browse the tracked runs
	.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db

clean: ## remove generated models/reports (keeps cached raw downloads)
	rm -rf models reports/permit_to_completion reports/application_to_issue
