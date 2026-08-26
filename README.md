# Midwest Solar Analytics

End-to-end ML pipeline that predicts how long a distributed solar PV project
takes, trained on real solar permit records from twelve municipal open-data
registers across eight Midwest states, plus utility interconnection records
from LBNL's *Tracking the Sun*.

XGBoost regression, tuned with Optuna under 5-fold cross-validation, tracked
in MLflow, explained with a SHAP attribution dashboard, and served from a
FastAPI prediction service with its own latency benchmark.

## What it predicts

| Model | Segment | Question |
|---|---|---|
| A | `permit_to_completion` | permit issued to installation complete / final inspection passed |
| B | `application_to_issue` | application filed to permit issued |

Model A is the "installation timeline" in the ordinary sense. Model B is the
permit-review window. They are trained and reported separately because they
are different quantities with different distributions; pooling them would
produce an R² driven mostly by which milestone pair a row belongs to rather
than by anything the model learned.

---

## Quick start

```bash
git clone <this repo> && cd midwest-solar-analytics
make venv                       # .venv from requirements.lock.txt + editable install
make test                       # the test suite, offline, no network
make smoke                      # whole pipeline end to end on a synthetic
                                # corpus, in a temp dir, no network
make data                       # download & assemble the real public corpus
```

`make smoke` is the fastest way to confirm the chain works: it runs baselines,
Optuna, the final fit, SHAP, serve, predict and the benchmark against a
generated corpus in a scratch directory, and touches nothing in the repository.
It is not a measurement; it is a proof that every stage still hands the next
one what it needs. CI runs exactly this.

Run the whole pipeline on the real corpus:

```bash
./scripts/run_pipeline.sh       # both segments; TRIALS=n to set the Optuna budget
```

Or one stage at a time:

```bash
make baselines SEGMENT=permit_to_completion   # four baselines on identical folds
make tune SEGMENT=permit_to_completion        # Optuna study (TRIALS=n to override)
make final SEGMENT=permit_to_completion       # tuned fit, quantile intervals, MLflow
make explain SEGMENT=permit_to_completion     # SHAP dashboard
make bench SEGMENT=permit_to_completion       # latency distribution
make mlflow-ui                                # browse the tracked runs
```

Serve and query:

```bash
make serve                      # uvicorn on :8000, docs at /docs
python -m scripts.demo_predict  # worked examples with SHAP explanations
```

---

## How it works

```
data/raw/                cached source payloads (gitignored)
src/msa/
  artifacts.py           the model-directory contract serving refuses to break
  ingest/                one module per register + shared Socrata/ArcGIS clients
    http.py              one timeout / retry / backoff policy for both clients
    build_dataset.py     concatenates every source onto the canonical schema
    tts.py               LBNL Tracking the Sun utility-record layer
  features/
    build.py             target construction, censoring guard, feature blocks
    store.py             training-time context snapshot for the serving path
  train/
    baseline.py          four baselines on identical folds
    tune.py              Optuna study, one MLflow run per trial
    final.py             tuned fit + quantile intervals + MLflow registration
    mlflow_model.py      the pyfunc flavour that is actually servable
  explain/shap_report.py TreeSHAP attributions -> self-contained HTML dashboard
  serve/
    app.py               FastAPI service
    schemas.py           request/response contracts, every field bounded
    logging_config.py    JSON logs + request-id correlation
  bench/latency.py       serial and concurrent latency measurement
scripts/smoke_pipeline.py  the whole chain, offline, in a scratch directory
.github/workflows/ci.yml   lint / tests / smoke / dependency audit
```

The pipeline runs in stages, each of which writes what the next one reads:

1. `msa.ingest.build_dataset` pulls every register, screens out non-PV work,
   maps each source's milestone dates onto the canonical
   `application_date` / `issue_date` / `completion_date` triple, and writes
   `data/processed/installations.parquet`.
2. `msa.train.baseline` scores four baselines (global median, group median,
   ridge, stock XGBoost) on the same 5-fold split the search will use.
3. `msa.train.tune` runs the Optuna study, minimising 5-fold CV MAE, with one
   MLflow run per trial. The study is persisted to `optuna.db` so an
   interrupted search resumes where it stopped.
4. `msa.train.final` fits the tuned model, fits a companion quantile booster
   for the prediction interval, conformalises that interval on a
   held-out-from-fit slice of the training window, scores the chronological
   holdout, snapshots the feature store, and registers the model with MLflow.
5. `msa.explain.shap_report` writes the SHAP dashboard.
6. `msa.serve.app` serves the artifact; `msa.bench.latency` measures it.

Design decisions and the reasoning behind them are in
[docs/METHODOLOGY.md](docs/METHODOLOGY.md). The short version:

* Every feature must be computable on the day the clock starts. Contractor
  history is an expanding mean over *strictly earlier* jobs; jurisdiction
  turnaround uses only permits *already closed*; the open-permit backlog uses
  only what is observable at time *t*. Five tests enforce this, including one
  that shuffles the target and requires the model to find nothing.
* Absolute time indices (`start_year`, days since the series began) are not
  features. A model that leans on an absolute date index has no defined
  behaviour on a permit dated beyond its training window, which is exactly
  the case a deployed model faces. `make ablate` fits the model both ways and
  writes the comparison to `reports/ablation_time_features.json`, so the
  trade-off can be inspected rather than assumed.
* The holdout is chronological *within each jurisdiction*. A single global
  cut-off would also change the mix of cities, confounding "generalises forward
  in time" with "generalises to a new city".
* Right-censoring is handled explicitly. Projects starting within one
  horizon of the snapshot are dropped, because only the fast ones could have
  finished in time to appear in the extract.

---

## API

```bash
curl -s -X POST 'http://127.0.0.1:8000/predict?explain=true' \
  -H 'content-type: application/json' -d '{
    "jurisdiction": "Minneapolis",
    "state": "MN",
    "start_date": "2025-04-15",
    "system_size_kw": 8.4,
    "reported_cost_usd": 24000,
    "sector": "residential",
    "contractor": "ALL ENERGY SOLAR"
  }'
```

Returns a point estimate, an 80% prediction interval (p10/p90 from a
companion multi-quantile booster, conformalised during training), the implied
completion date, and the ranked SHAP attributions for that single prediction.
The point estimate always lies inside the interval.

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness; model identity, version, training date and window |
| `GET /ready` | readiness; 200 only if the model is loaded and just scored a canary row, else 503 |
| `GET /metadata` | full model card: metrics, features, feature-store date |
| `POST /predict` | one project; `?explain=true` adds SHAP attributions |
| `POST /predict/batch` | up to 1000 projects per call; `?explain=true` is accepted for batches of 50 or fewer and answers 422 above that |
| `GET /metrics` | in-process request counters and latency percentiles |

Only `jurisdiction` and `start_date` are required: open-permit records are
routinely incomplete, and XGBoost consumes missing numerics natively.

---

## SHAP dashboard

`make explain` writes a self-contained `reports/<segment>/shap_dashboard.html`
containing the global beeswarm, the mean-|SHAP| ranking, three worked
single-prediction waterfalls (a fast job, a typical job, a slow job), dependence
plots for the top drivers, and a per-jurisdiction driver breakdown. Raw
attributions are also written to `shap_values.parquet` so a stakeholder can join
them onto their own permits.

Attributions come from `shap.TreeExplainer`, which computes exact Shapley
values on a gradient-boosted tree, not an approximation.

---

## Running it in production

The service is built to fail loudly rather than answer wrongly. Each behaviour
below is asserted by a test in `tests/test_serve_hardening.py`, and the startup
cases were also exercised against a real `uvicorn` process.

Startup. `models/<segment>/` is validated against an explicit artifact
contract (`src/msa/artifacts.py`) before the process will serve anything:
schema version, the exact feature list, categorical levels for every
categorical column, the booster's own feature count, and a canary prediction
that must return a finite number. If any check fails the process refuses to
start: `uvicorn` logs the reason and exits 3 without binding the socket.

```bash
$ MSA_MODELS_DIR=/tmp/empty uvicorn msa.serve.app:app
CRITICAL msa.serve.app: refusing to start: .../metadata.json not found;
         run `python -m msa.train.final --segment permit_to_completion` first.
ERROR    uvicorn.error: Application startup failed. Exiting.
$ echo $?
3
```

Verified this way for a missing artifact, a truncated booster and an
incompatible `artifact_schema_version`. Set `MSA_ALLOW_DEGRADED_START=1` to
start anyway: `/health` then answers 200 with `"status": "degraded"`, `/ready`
returns 503, and every prediction endpoint returns 503 with a short reason.
There is no configuration in which the service starts and quietly serves
numbers from an artifact it could not verify.

Why a schema version matters. A model fitted before a feature was added
still loads into XGBoost and still returns floats: floats that answer a
different question. The feature-list check is what turns that into a startup
failure instead of a silent wrong answer.

Versioning. `model_version` is `<segment>-<YYYYMMDD>-<sha256[:12] of
model.json>`: content-bound, so it changes if a single byte of the booster
changes and is stable across a copy. It appears on `/health`, `/ready`,
`/metrics` and on every prediction, so any answer can be traced back to the
artifact that produced it.

Validation. Every field is bounded. Strings are length-capped, numerics are
range-capped, dates must be plausible, unknown fields are rejected
(`extra="forbid"`), and `Infinity`/`NaN` (which Python's JSON parser accepts,
and which sail straight through a `ge=0` bound) are rejected outright. Batches
are capped at 1000 items and bodies at 4 MiB (413), enforced by counting
bytes as they arrive, so a `Transfer-Encoding: chunked` upload that declares no
length is bounded too. Malformed single and batch payloads are asserted to
return 422, never 500.

Errors. Responses carry `{error, detail, request_id}` and nothing else. No
exception text, no traceback, no filesystem path; a test greps every error
path for those patterns. The detail goes to the log instead, keyed by the same
`request_id`.

Observability. One JSON object per log line, with the request id folded in
via a `contextvars` variable so it reaches library log records too. `X-Request-ID`
is accepted from the caller (length-capped) or generated, and echoed back.

Concurrency. The booster is loaded once and shared; XGBoost's prediction
functions are [documented thread-safe](https://xgboost.readthedocs.io/en/stable/prediction.html)
for `gbtree` as long as nothing mutates the model, which nothing here does.
TreeSHAP is bounded to `MSA_MAX_CONCURRENT_EXPLAIN` (default 2) concurrent
calls so peak memory is a constant rather than a function of how many clients
arrive at once. The service is CPU-bound in the booster, so extra concurrency
buys queueing rather than work; scale by adding workers. `make bench` writes
the measured distribution to `reports/latency_<mode>_<segment>.json`.

Environment variables. `MSA_SERVE_SEGMENT`, `MSA_MODELS_DIR`,
`MSA_ALLOW_DEGRADED_START`, `MSA_MAX_BODY_BYTES`, `MSA_MAX_CONCURRENT_EXPLAIN`,
`MSA_LOG_LEVEL`, `MSA_LOG_FORMAT`. No secrets are read at request time; see
[docs/SECURITY.md](docs/SECURITY.md).

## Licence / attribution

All data is municipal, state or federal open data. *Tracking the Sun* is
public data from Lawrence Berkeley National Laboratory.
