# Midwest Solar Analytics

End-to-end ML pipeline that predicts how long a distributed solar PV project
takes, trained on **29,425 real solar permit records** scraped from twelve
municipal open-data registers across eight Midwest states, plus utility
interconnection records from LBNL's *Tracking the Sun*.

XGBoost regression · Optuna (200 trials, 5-fold CV) · MLflow tracking ·
SHAP attribution dashboard · FastAPI prediction service with a latency benchmark.

> **Every number in this README is machine-generated from artifacts in
> `reports/` by `python -m scripts.make_results_tables`.** Nothing is typed in
> by hand, so a figure cannot drift from what was actually measured, and a claim
> whose artifact does not exist is printed as **NOT MEASURED** rather than
> quietly omitted.
>
> **Current status:** the data pipeline, the feature layer, the four baselines,
> the serving stack and the test suite are built and measured. The 200-trial
> Optuna study was **not run to completion** in this working tree, so the tuned
> model, the MLflow run export, the latency distribution and the SHAP dashboard
> are unmeasured — see the scorecard below and
> **[Honest assessment](#honest-assessment)**. `./scripts/run_pipeline.sh`
> produces all of it.

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
make venv                       # .venv + pinned deps + editable install
make data                       # download & assemble the public corpus (~1 min)
make test                       # 53 tests, offline
TRIALS=20 make pipeline         # smoke run of everything, ~10 min
```

Full reproduction (what produced the numbers below):

```bash
./scripts/run_pipeline.sh       # both segments, 200 Optuna trials each
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
| R² (model A, 5-fold CV) | 0.86 | not measured — run `make tune final SEGMENT=permit_to_completion` | **NOT MEASURED** |
| MAE (model A, 5-fold CV) | 4.2 days | not measured — run `make tune final SEGMENT=permit_to_completion` | **NOT MEASURED** |
| improvement over baseline (model A) | 23% | not measured — run `make tune final SEGMENT=permit_to_completion` | **NOT MEASURED** |
| R² (model B, 5-fold CV) | 0.86 | not measured — run `make tune final SEGMENT=application_to_issue` | **NOT MEASURED** |
| MAE (model B, 5-fold CV) | 4.2 days | not measured — run `make tune final SEGMENT=application_to_issue` | **NOT MEASURED** |
| improvement over baseline (model B) | 23% | not measured — run `make tune final SEGMENT=application_to_issue` | **NOT MEASURED** |
| MLflow runs | 60+ | 145 (136 optuna, 8 baseline, 1 optuna-parent) | **MET** |
| p95 inference latency | < 50 ms | not measured — run `make bench` | **NOT MEASURED** |


### Headline metrics (tuned models)

> **Not measured.** The 200-trial Optuna study was stopped before
> completing, so no tuned model was fitted, no holdout was scored,
> and there is no latency distribution or SHAP dashboard. Rather
> than print placeholder numbers this section stays empty until
> `./scripts/run_pipeline.sh` has produced the artifacts, at which
> point `python -m scripts.make_results_tables` fills it in.

What **is** measured is the baseline layer below — four models per
segment on identical 5-fold splits, which is what the tuned model
must be judged against.

### Partial Optuna study (incomplete)

#### `permit_to_completion` — INCOMPLETE study, 136 of 200 trials logged

> These are real logged trials from a study that was **stopped before
> completing**. No final model was fitted from them and no holdout was
> scored, so they are *not* the project's results — they are a partial
> read on where a finished study would land.

| | value |
|---|---:|
| trials logged | 136 of 200 |
| best trial CV MAE (days) | 38.476 |
| best trial CV R² | 0.3364 |
| stock XGBoost CV MAE (days) | 43.419 |
| reduction in CV MAE so far | 11.4% |


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
  ingest/                one module per register + shared Socrata/ArcGIS clients
    build_dataset.py     concatenates every source onto the canonical schema
    tts.py               LBNL Tracking the Sun utility-record layer
  features/
    build.py             target construction, censoring guard, feature blocks
    store.py             training-time context snapshot for the serving path
  train/
    baseline.py          four baselines on identical folds
    tune.py              200-trial Optuna study, one MLflow run per trial
    final.py             tuned fit + quantile intervals + MLflow registration
  explain/shap_report.py TreeSHAP attributions -> self-contained HTML dashboard
  serve/app.py           FastAPI service
  bench/latency.py       latency measurement
```

Key design decisions, with the evidence behind them, are in
**[docs/METHODOLOGY.md](docs/METHODOLOGY.md)**. The short version:

* **Every feature must be computable on the day the clock starts.** Contractor
  history is an expanding mean over *strictly earlier* jobs; jurisdiction
  turnaround uses only permits *already closed*; the open-permit backlog uses
  only what is observable at time *t*. Five tests enforce this, including one
  that shuffles the target and requires R² < 0.05.
* **Absolute time indices are excluded.** `start_year` and days-since-start let
  the booster memorise each period's level, which inflates random-fold CV and
  collapses on a chronological holdout. `make ablate` quantifies the effect on
  the current corpus; no figure is quoted here because that artifact has not
  been generated in this tree.
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
| `GET /health` | model identity, training window, jurisdictions |
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

## Honest assessment

### What is measured, and what is not

| Component | Status |
|---|---|
| 29,425-record Midwest solar permit corpus, 12 jurisdictions, 8 states | **measured** — `data/processed/dataset_summary.json` |
| Horizon coverage / censoring analysis | **measured** — `reports/horizon_coverage.json` |
| Four baselines per segment, identical 5-fold splits | **measured** — `reports/<segment>/baseline_results.json` |
| Test suite (53 tests, incl. 5 leakage tests) | **measured** — `make test` |
| MLflow tracking — 145 logged runs exported | **measured** — `reports/mlflow_runs_summary.json` |
| Tracking the Sun state coverage, all 7 editions | **measured** — `reports/tts_state_coverage.json` |
| 200-trial Optuna study (stopped at 136/200), tuned model, latency p95, SHAP dashboard | **not measured in this tree** — `./scripts/run_pipeline.sh` |

The study was started three times and stopped three times: twice deliberately
(a leakage bug and a look-ahead bug were found mid-run and the affected trials
had to be discarded), once by a crash at trial 185 of 200 when a `log1p`-target
booster produced a score whose `expm1` overflowed to `+inf`. All three causes
are fixed and pinned by tests; the study is now also resumable
(`sqlite:///optuna.db`) so a future interruption costs minutes rather than an
hour. It was then stopped a fourth time on request, before completing.

### What the baselines already tell us about the résumé's numbers

These are real measurements, and they set the range the tuned model will land in:

* **Model B** (permit review, application → issue): stock XGBoost gets
  **CV MAE 6.11 days, CV R² 0.437**. Tuning moves numbers like this by single-digit
  percent, so a **4.2-day MAE is plausibly reachable; R² = 0.86 is not.**
* **Model A** (installation timeline, permit → completion): stock XGBoost gets
  **CV MAE 43.4 days, CV R² 0.311**. This target is dominated by administrative
  permit-closure lag, and **neither 4.2 days nor R² = 0.86 is reachable** — it is
  off by an order of magnitude on MAE.

The 136 Optuna trials that *were* logged before the study was stopped sharpen
this further. The best trial reached **CV MAE 38.48, CV R² 0.336** — a **11.4%**
reduction in CV MAE over stock XGBoost. The résumé claims **23%**. Tuning has
already spent two-thirds of its trial budget to get half that, so **the 23%
figure is unlikely to be met against a stock-XGBoost baseline**, though it is
comfortably exceeded against the naive median baseline (53.30 → 38.48 is 27.8%).
Which baseline the "23%" refers to is exactly the ambiguity the results table
now resolves by reporting the improvement against all four.

So on current evidence the résumé's **R² = 0.86 will not be met by either
model**, and the **4.2-day MAE is only meaningful for the permit-review
segment**, not for the installation timeline the line describes. The scorecard
above will record whichever verdict the completed study produces.

### Known weaknesses

* **The target is noisy by nature.** A permit's "completion" date is when an
  inspector closed the record, which can lag physical completion by months. Much
  of the residual is administrative, not physical, and no feature set fixes that.
* **Stock XGBoost overfits the chronological holdout badly** (Model A holdout
  R² = −0.52 against CV R² = 0.31). That gap is precisely what the Optuna study
  exists to close, and it is unclosed until the study is run.
* **Jurisdiction coverage is lopsided.** Chicago is 44% of the corpus and
  Bismarck contributes 16 records; the small registers cannot support
  per-jurisdiction conclusions.
* **Solar is identified from free-text scope of work** in most registers. The
  matcher is recall-first with an explicit non-PV veto list, which is the right
  trade-off here, but it is not a coded work class in most cities.
* **Tracking the Sun coverage ends 2024-03** and covers only IL/MN/OH/WI, so its
  context features saturate for recent permits in other states.
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

`mlflow.db` and `mlartifacts/` are gitignored; `reports/mlflow_runs.csv` and
`reports/mlflow_runs_summary.json` are committed as the evidence trail.

Browse the tracked runs with `make mlflow-ui`.

---

## Licence / attribution

All data is municipal, state or federal open data. *Tracking the Sun* is
public data from Lawrence Berkeley National Laboratory.
