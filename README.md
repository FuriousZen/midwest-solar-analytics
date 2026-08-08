# Midwest Solar Analytics

End-to-end ML pipeline that predicts how long a distributed solar PV project
takes, trained on **29,425 real solar permit records** scraped from twelve
municipal open-data registers across eight Midwest states, plus utility
interconnection records from LBNL's *Tracking the Sun*.

XGBoost regression · Optuna (200 trials, 5-fold CV) · MLflow tracking ·
SHAP attribution dashboard · FastAPI prediction service with measured p95 latency.

> **All numbers in this README are measured, not aspirational.**
> Every figure is generated from artifacts in `reports/` by
> `python -m scripts.make_results_tables`, and each has a command you can run to
> reproduce it. Where the results fall short of what was hoped for, they say so.
> See **[docs/RESULTS.md](docs/RESULTS.md)** for the full tables and
> **[Honest assessment](#honest-assessment)** below for the shortfalls.

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
make test                       # 49 tests, offline
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
  the booster memorise each period's level: measured effect on the
  `application_to_issue` chronological holdout was R² −0.10 → +0.17 when removed.
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

<!-- HONEST_ASSESSMENT -->

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
