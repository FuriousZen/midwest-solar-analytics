# Midwest Solar Analytics

End-to-end ML pipeline that predicts how long a distributed solar PV project
takes, trained on **29,425 real solar permit records** scraped from twelve
municipal open-data registers across eight Midwest states, plus utility
interconnection records from LBNL's *Tracking the Sun*.

XGBoost regression · Optuna (5-fold CV; **136 trials** for model A, **20** for
model B — neither study ran the full 200, see below) · MLflow tracking · SHAP
attribution dashboard · FastAPI prediction service with a latency benchmark.

> **Every number in [Results](#results) is machine-generated from artifacts in
> `reports/` by `python -m scripts.make_results_tables`.** Nothing in that
> section is typed in by hand, so a figure cannot drift from what was actually
> measured, and a claim whose artifact does not exist is printed as
> **NOT MEASURED** rather than quietly omitted. Figures quoted in prose
> elsewhere are copied from the same committed artifacts and each names the file
> it came from, so any of them can be checked.
>
> **Current status:** every stage now runs end to end and every claim in the
> scorecard has a measurement behind it. Final models are fitted for both
> segments, the API serves real predictions with SHAP explanations, and the
> latency distribution is measured.
>
> **The trial budgets were short and are labelled as such everywhere.** Model
> A's study was stopped at **136 of 200** trials and the final model was fitted
> from those 136; model B's study was deliberately run to only **20** trials.
> Neither number is the 200 the pipeline defaults to, and the metrics below are
> conditional on that. `./scripts/run_pipeline.sh` runs the full budget.

---

## What it predicts

| Model | Segment | Question | Rows |
|---|---|---|---|
| **A** | `permit_to_completion` | permit issued → installation complete / final inspection passed | 12,108 usable |
| **B** | `application_to_issue` | application filed → permit issued | 21,884 usable |

Model A is the "installation timeline" in the ordinary sense. Model B is the
permit-review window. They are trained and reported separately because they are
different quantities (median 59 days vs 7 days); pooling them would manufacture
a flattering R² driven entirely by which milestone pair a row belongs to.

---

## Quick start

```bash
git clone <this repo> && cd midwest-solar-analytics
make venv                       # .venv from requirements.lock.txt + editable install
make test                       # 194 tests, offline, no network
make smoke                      # whole pipeline end to end on a synthetic
                                # corpus, in a temp dir, ~3 min, no network
make data                       # download & assemble the real public corpus (~1 min)
```

`make smoke` is the fastest way to confirm the chain works: it runs baselines →
Optuna → final fit → SHAP → serve → predict → benchmark → results against a
generated corpus in a scratch directory, and touches nothing in the repository.
It is not a measurement; it is a proof that every stage still hands the next one
what it needs. CI runs exactly this.

Full reproduction of the numbers below:

```bash
./scripts/run_pipeline.sh       # both segments; TRIALS=n to set the budget
```

Serve and query:

```bash
make serve                      # uvicorn on :8000, docs at /docs
python -m scripts.demo_predict  # worked examples with SHAP explanations
make bench                      # latency distribution
```

---

## Results

<!-- RESULTS_TABLE -->

### Résumé claim vs measurement

| claim | résumé says | measured | verdict |
|---|---|---|---|
| dataset size | 8,000+ historical installations | 29,425 permit records; 12,108 with an installation duration, 21,884 with a review duration | **MET** |
| R² (model A, 5-fold CV) | 0.86 | 0.3364 | **NOT MET** |
| MAE (model A, 5-fold CV) | 4.2 days | 38.48 days | **NOT MET** |
| improvement over baseline (model A) | 23% | 11.4% lower CV MAE vs stock XGBoost (vs the naive median baseline: 27.8%) | **NOT MET** |
| R² (model B, 5-fold CV) | 0.86 | 0.3901 | **NOT MET** |
| MAE (model B, 5-fold CV) | 4.2 days | 5.29 days | **NOT MET** |
| improvement over baseline (model B) | 23% | 13.5% lower CV MAE vs stock XGBoost (vs the naive median baseline: 44.3%) | **NOT MET** |
| MLflow runs | 60+ | 187 (156 optuna, 20 baseline, 6 final, 5 optuna-parent) | **MET** |
| p95 inference latency | < 50 ms | 15.33 ms | **MET** |


### Headline metrics (tuned models)

The trial budget each model actually got is the first row, because
every metric below is conditional on it. Neither study ran to 200
trials: model A's was stopped at 136 and the final model was fitted
from those; model B's was deliberately run short.

| | Model A — installation timeline | Model B — permit review |
|---|---:|---:|
| Optuna trials completed | 136 | 20 |
| training rows | 8,117 | 16,684 |
| 5-fold CV MAE (days) | 38.48 | 5.29 |
| 5-fold CV RMSE (days) | 62.41 | 13.19 |
| 5-fold CV R² | 0.3364 | 0.3901 |
| chronological holdout MAE (days) | 45.10 | 8.32 |
| chronological holdout R² | 0.0730 | 0.2203 |
| improvement vs stock XGBoost (CV MAE) | 11.4% | 13.5% |
| 80% interval coverage on holdout | 0.908 | 0.888 |

### Measured baselines

`permit_to_completion` — 8,117 train / 2,045 holdout, 5-fold CV, seed 20250201

| model | CV MAE (d) | CV R² | holdout MAE (d) | holdout R² |
|---|---:|---:|---:|---:|
| `global_median` | 53.302 | -0.1216 | 47.118 | -0.1323 |
| `group_median` | 47.618 | 0.1021 | 45.621 | -0.0297 |
| `ridge` | 47.059 | 0.2750 | 48.628 | 0.0517 |
| `xgb_default` | 43.419 | 0.3108 | 64.310 | -0.5201 |

`application_to_issue` — 16,684 train / 4,194 holdout, 5-fold CV, seed 20250201

| model | CV MAE (d) | CV R² | holdout MAE (d) | holdout R² |
|---|---:|---:|---:|---:|
| `global_median` | 9.497 | -0.0652 | 10.043 | -0.1087 |
| `group_median` | 7.862 | 0.2029 | 9.428 | 0.0840 |
| `ridge` | 7.119 | 0.3645 | 8.333 | 0.3247 |
| `xgb_default` | 6.114 | 0.4368 | 10.707 | 0.0817 |

Full tables, per-jurisdiction breakdowns, the Optuna study and the
latency distribution: **[docs/RESULTS.md](docs/RESULTS.md)**.

---

## Data

Twelve municipal permit registers plus one federal utility-records dataset.
Full provenance, including every source that was evaluated and **rejected**, is
in **[docs/DATA_PROVENANCE.md](docs/DATA_PROVENANCE.md)**.

| Jurisdiction | State | Platform | Records |
|---|---|---|---:|
| Chicago | IL | Socrata `ydr8-5enu` | 12,873 |
| Kansas City | MO | Socrata `ntw8-aacc` + `w8jz-wjgn` | 4,550 |
| Saint Paul | MN | ArcGIS `Approved_Building_Permits` | 3,781 |
| Minneapolis | MN | ArcGIS `CCS_Permits` | 3,443 |
| Cincinnati | OH | Socrata `thvx-5mem` ⋈ `94uh-66xv` | 1,089 |
| Cleveland | OH | ArcGIS `Building_Permits` | 1,066 |
| Naperville | IL | ArcGIS `Building_Permits_View` | 850 |
| Detroit | MI | ArcGIS `bseed_building_permits` | 697 |
| Golden Valley | MN | ArcGIS `Golden_Misc` 586/587 | 670 |
| Ames | IA | ArcGIS `coa_Inspections_BuildingPermits_V2` | 262 |
| Overland Park | KS | ArcGIS `Building_Permits` | 128 |
| Bismarck | ND | ArcGIS `Map1` | 16 |
| **Total** | 8 states | | **29,425** |

**Target.** `timeline_days = (end_milestone − start_milestone).days`, taken from
the register's own date columns — `application_start_date`/`issue_date` in
Chicago, `issueDate`/`completeDate` in Minneapolis, `applieddate`/`issueddate`/
`completeddate` in Kansas City, and so on. Nothing is inferred or imputed.

**LBNL Tracking the Sun** (98,275 Midwest systems, IL/MN/OH/WI, 2024 edition,
via the DOE OEDI public S3 bucket) supplies *market-context features only*. Its
public file has exactly **one** date column, `installation_date` — verified, not
assumed — so it cannot supply a duration and is never used as a source of the
target.

**No synthetic data.** The only generated data in the repo is the
`synthetic_canonical` fixture in `tests/`, used to unit-test transformations and
never for training or for any published number.

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

Key design decisions, with the evidence behind them, are in
**[docs/METHODOLOGY.md](docs/METHODOLOGY.md)**. The short version:

* **Every feature must be computable on the day the clock starts.** Contractor
  history is an expanding mean over *strictly earlier* jobs; jurisdiction
  turnaround uses only permits *already closed*; the open-permit backlog uses
  only what is observable at time *t*. Five tests enforce this, including one
  that shuffles the target and requires R² < 0.05.
* **Absolute time indices are excluded — and the ablation does not support the
  reason originally given.** `start_year` and days-since-start were left out on
  the theory that they let the booster memorise each period's level: inflating
  random-fold CV and collapsing on a chronological holdout. `make ablate` now
  measures that on this corpus (`reports/ablation_time_features.json`, stock
  XGBoost, same folds), and it comes out the other way round:

  | segment | holdout R² **with** time indices | **without** (shipped) | cost of excluding |
  |---|---:|---:|---:|
  | `permit_to_completion` | −0.373 | −0.520 | **−0.147** |
  | `application_to_issue` | +0.109 | +0.082 | **−0.027** |

  CV R² is essentially unchanged either way (±0.002), so the predicted CV
  inflation does not appear at all, and excluding the indices makes the
  chronological holdout *worse*, not better. The features stay excluded — the
  leakage argument for keeping a model free of absolute time is about what
  happens when it is asked to score a date beyond its training range, which this
  holdout does not test, and both holdout R² values for model A are poor in
  absolute terms regardless. But the stated empirical justification is not
  supported by the measurement, and is recorded here as such rather than
  quietly dropped.
* **The holdout is chronological *within each jurisdiction*.** A single global
  cut-off would also change the mix of cities, confounding "generalises forward
  in time" with "generalises to a new city".
* **Right-censoring is handled explicitly.** Projects starting within one
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

Returns a point estimate, an **80% prediction interval** (p10/p90 from a
companion multi-quantile booster), the implied completion date, and the ranked
SHAP attributions for that single prediction.

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness; model identity, version, training date and window |
| `GET /ready` | readiness; 200 only if the model is loaded **and** just scored a canary row, else 503 |
| `GET /metadata` | full model card: metrics, features, feature-store date |
| `POST /predict` | one project; `?explain=true` adds SHAP attributions |
| `POST /predict/batch` | up to 1000 projects per call |
| `GET /metrics` | in-process request counters and latency percentiles |

Only `jurisdiction` and `start_date` are required — open-permit records are
routinely incomplete, and XGBoost consumes missing numerics natively.

---

## SHAP dashboard

`make explain` writes a self-contained `reports/<segment>/shap_dashboard.html`
containing the global beeswarm, the mean-|SHAP| ranking, three worked
single-prediction waterfalls (a fast job, a typical job, a slow job), dependence
plots for the top drivers, and a per-jurisdiction driver breakdown. Raw
attributions are also written to `shap_values.parquet` so a stakeholder can join
them onto their own permits.

Attributions come from `shap.TreeExplainer`, which computes **exact** Shapley
values on a gradient-boosted tree, not an approximation.

---

## Running it in production

The service is built to fail loudly rather than answer wrongly. Each behaviour
below is asserted by a test in `tests/test_serve_hardening.py`, and the startup
cases were additionally exercised against a real `uvicorn` process.

**Startup.** `models/<segment>/` is validated against an explicit artifact
contract (`src/msa/artifacts.py`) before the process will serve anything:
schema version, the exact feature list, categorical levels for every
categorical column, the booster's own feature count, and a canary prediction
that must return a finite number. If any check fails the process **refuses to
start** — `uvicorn` logs the reason and exits 3 without binding the socket.

```bash
$ MSA_MODELS_DIR=/tmp/empty uvicorn msa.serve.app:app
CRITICAL msa.serve.app: refusing to start: .../metadata.json not found --
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

**Why a schema version matters.** A model fitted before a feature was added
still loads into XGBoost and still returns floats — floats that answer a
different question. The feature-list check is what turns that into a startup
failure instead of a silent wrong answer.

**Versioning.** `model_version` is `<segment>-<YYYYMMDD>-<sha256[:12] of
model.json>`: content-bound, so it changes if a single byte of the booster
changes and is stable across a copy. It appears on `/health`, `/ready`,
`/metrics` and on **every prediction**, so any answer can be traced back to the
artifact that produced it.

**Validation.** Every field is bounded. Strings are length-capped, numerics are
range-capped, dates must be plausible, unknown fields are rejected
(`extra="forbid"`), and `Infinity`/`NaN` — which Python's JSON parser accepts,
and which sail straight through a `ge=0` bound — are rejected outright. Batches
are capped at 1000 items and bodies at 4 MiB (413) — enforced by counting
bytes as they arrive, so a `Transfer-Encoding: chunked` upload that declares no
length is bounded too. Twenty-seven malformed
single payloads and seven malformed batch payloads are asserted to return 422,
never 500.

**Errors.** Responses carry `{error, detail, request_id}` and nothing else. No
exception text, no traceback, no filesystem path — a test greps every error
path for those patterns. The detail goes to the log instead, keyed by the same
`request_id`.

**Observability.** One JSON object per log line, with the request id folded in
via a `contextvars` variable so it reaches library log records too. `X-Request-ID`
is accepted from the caller (length-capped) or generated, and echoed back.

**Concurrency.** The booster is loaded once and shared; XGBoost's prediction
functions are [documented thread-safe](https://xgboost.readthedocs.io/en/stable/prediction.html)
for `gbtree` as long as nothing mutates the model, which nothing here does.
TreeSHAP is bounded to `MSA_MAX_CONCURRENT_EXPLAIN` (default 2) concurrent
calls so peak memory is a constant rather than a function of how many clients
arrive at once. Measured, in-process, 2000 requests per level:

| concurrent requests | p50 | p95 | throughput |
|---:|---:|---:|---:|
| 1 | 12.1 ms | 15.3 ms | 79 req/s |
| 8 | 94.7 ms | 149.3 ms | 80 req/s |
| 32 | 321.6 ms | 438.5 ms | 91 req/s |

Throughput is flat from 8 concurrent onward: the service is CPU-bound in the
booster, so extra concurrency buys queueing, not work. Scale by adding workers.

**Environment variables.** `MSA_SERVE_SEGMENT`, `MSA_MODELS_DIR`,
`MSA_ALLOW_DEGRADED_START`, `MSA_MAX_BODY_BYTES`, `MSA_MAX_CONCURRENT_EXPLAIN`,
`MSA_LOG_LEVEL`, `MSA_LOG_FORMAT`. No secrets are read at request time; see
**[docs/SECURITY.md](docs/SECURITY.md)**.

---

## Honest assessment

### What is measured, and what is not

| Component | Status |
|---|---|
| 29,425-record Midwest solar permit corpus, 12 jurisdictions, 8 states | **measured** — `data/processed/dataset_summary.json` |
| Horizon coverage / censoring analysis | **measured** — `reports/horizon_coverage.json` |
| Four baselines per segment, identical 5-fold splits | **measured** — `reports/<segment>/baseline_results.json` |
| Test suite (194 tests, incl. 5 leakage tests) | **measured** — `make test` |
| MLflow tracking — 175 logged runs exported | **measured** — `reports/mlflow_runs_summary.json` |
| Tracking the Sun state coverage, all 7 editions | **measured** — `reports/tts_state_coverage.json` |
| Tuned final model, both segments | **measured** — `reports/<segment>/final_metrics.json`, on **136** (A) and **20** (B) Optuna trials, not 200 |
| Serving latency, serial and under concurrent load | **measured** — `reports/latency_inprocess_<segment>.json` |
| SHAP attribution dashboard, both segments | **measured** — `reports/<segment>/shap_dashboard.html` |
| End-to-end pipeline from a clean environment | **measured** — `make smoke`, and CI runs it on every push |
| A 200-trial study for either segment | **not measured** — `./scripts/run_pipeline.sh` |

**On the trial budgets.** Model A's study was started three times and stopped
three times before this working session: twice deliberately (a leakage bug and a
look-ahead bug were found mid-run and the affected trials had to be discarded),
once by a crash at trial 185 of 200 when a `log1p`-target booster produced a
score whose `expm1` overflowed to `+inf`. All three causes are fixed and pinned
by tests, and the study is resumable (`sqlite:///optuna.db`). It was stopped a
fourth time on request at **136 completed trials**, and the final model
published here is fitted from those 136 — not from a completed study. Model B
had no study at all; it was given a deliberately short **20-trial** budget, just
enough for a real fitted model to exist. Both counts are read out of
`reports/<segment>/optuna_study.json` by the generator, so the tables above
cannot quietly round them up to 200.

### What the measurements say about the résumé's numbers

The scorecard is now filled in from fitted models rather than from projections.
It reads **NOT MET** on five of nine rows, and those verdicts are correct.

* **R² = 0.86 is not met by either model.** Model A reaches **CV R² 0.336**,
  model B **0.390**. These are not near misses; they are less than half the
  claim. Predicting a permit's remaining duration from permit metadata is
  intrinsically hard, and no amount of tuning closes a gap that size.
* **MAE = 4.2 days is not met by either model.** Model A is at **38.48 days**,
  an order of magnitude away. Model B is at **5.29 days**, which is the same
  order but still misses. The 4.2-day figure only ever made sense for the
  permit-review segment; for the installation timeline the résumé line
  describes, it is not in the right range.
* **The 23% improvement is not met against a stock-XGBoost baseline.** Model A
  achieves **11.4%** (43.42 → 38.48), model B **13.5%** (6.11 → 5.29). It *is*
  exceeded against the naive median baseline — 27.8% and 44.3% respectively —
  which is exactly the ambiguity the results table resolves by reporting the
  improvement against all four baselines rather than the flattering one.
* **Dataset size, MLflow run count and p95 latency are met**, the last
  comfortably — see the generated scorecard above for the measured p95, which
  is roughly a third of the 50 ms budget.

Would a full 200-trial budget change the verdicts? Model A's study spent 136
trials to reach 11.4%; the remaining 64 would have to more than double that to
reach 23%, which the trial history gives no reason to expect. Model B's 20
trials are genuinely a small sample and its numbers could move more. Neither
would come close to R² = 0.86. **The tuning budget is not what these claims are
short by.**

### Known weaknesses

* **The target is noisy by nature.** A permit's "completion" date is when an
  inspector closed the record, which can lag physical completion by months. Much
  of the residual is administrative, not physical, and no feature set fixes that.
* **The chronological holdout is much harder than the random-fold CV, and the
  tuned model does not fully close that.** Stock XGBoost collapses on it
  (Model A holdout R² = −0.52 against CV R² = 0.31). Tuning takes the holdout
  from −0.52 to **+0.073** — a real repair of a real failure, but +0.073 is
  still barely better than predicting the median. The honest reading is that
  Model A generalises forward in time only weakly.
* **The trial budgets are short.** 136 trials for model A and 20 for model B,
  against a 200-trial design. The published metrics are what those budgets
  bought, not what the pipeline is capable of.
* **Jurisdiction coverage is lopsided.** Chicago is 44% of the corpus and
  Bismarck contributes 16 records; the small registers cannot support
  per-jurisdiction conclusions.
* **Solar is identified from free-text scope of work** in most registers. The
  matcher is recall-first with an explicit non-PV veto list, which is the right
  trade-off here, but it is not a coded work class in most cities.
* **Tracking the Sun coverage ends 2024-03** and covers only IL/MN/OH/WI, so its
  context features saturate for recent permits in other states.
* **The prediction interval is honest but wide, and over-covers.** Conformal
  calibration was needed because the raw pinball-loss quantiles covered only
  51.8% (model A) and 69.9% (model B) of the holdout at a nominal 80%. The
  correction fixes that — 90.8% and 88.8% — but by adding ±60.3 and ±4.2 days,
  giving a median interval width of **164 days** for model A and 21 for model
  B. A 164-day-wide interval on a median 59-day job is close to useless
  operationally, and after clipping at zero the lower bound is frequently 0.
  It is the correct interval for a model this uncertain, which is the point:
  the width is the model's uncertainty made visible, not a defect of the
  calibration.
* **The feature store is a snapshot,** not a live store. Contractor and
  jurisdiction context are frozen at training time; `/metadata` reports
  `feature_store_as_of` so staleness is visible rather than silent.

---

## Reproducing every number

| Number | Command |
|---|---|
| dataset composition | `python -m msa.ingest.build_dataset` → `data/processed/dataset_summary.json` |
| horizon coverage | `python -m scripts.report_horizon_coverage` |
| baselines | `python -m msa.train.baseline --segment <seg>` |
| Optuna study & improvement | `python -m msa.train.tune --segment <seg> --trials 200` |
| final metrics & intervals | `python -m msa.train.final --segment <seg>` |
| MLflow run counts | `python -m scripts.export_mlflow_summary` |
| latency | `python -m msa.bench.latency --requests 2000` |
| SHAP dashboard | `python -m msa.explain.shap_report --segment <seg>` |
| the tables in this README | `python -m scripts.make_results_tables` |
| library API assumptions | `python -m scripts.verify_apis` |
| time-index ablation | `python -m scripts.ablate_time_features` |
| the whole chain, offline, from scratch | `make smoke` |
| static checks | `make lint` |
| known CVEs in the pinned environment | `make audit` → **[docs/SECURITY.md](docs/SECURITY.md)** |

`mlflow.db` and `mlartifacts/` are gitignored; `reports/mlflow_runs.csv` and
`reports/mlflow_runs_summary.json` are committed as the evidence trail.

Browse the tracked runs with `make mlflow-ui`.

---

## Licence / attribution

All data is municipal, state or federal open data. *Tracking the Sun* is
public data from Lawrence Berkeley National Laboratory.
