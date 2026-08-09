# Methodology

## The prediction problem

Given a solar PV project at the moment its clock starts, predict how many days
until the next milestone. Two models are trained by the same pipeline:

| Model | Segment | Question it answers |
|---|---|---|
| **A** | `permit_to_completion` | *"The permit is issued. How long until this installation is finished and signed off?"* |
| **B** | `application_to_issue` | *"The application is filed. How long until the permit is approved?"* |

Model A is the "installation timeline" in the ordinary sense. Model B is the
permit-review window. They are reported separately because they are different
quantities with different distributions (median 59 days vs 7 days) and pooling
them would produce a flattering R² driven entirely by which milestone pair a
row belongs to.

## The leakage rule

**Every feature must be computable from information available on the day the
clock starts.** Nothing may depend on the record's own `end_date`, or on the
`end_date` of any record that had not yet finished.

That rule is what makes several of the strongest features legitimate:

* `contractor_prior_mean_days` — the mean duration of that contractor's
  **strictly earlier** jobs. An expanding mean with `shift(1)`, so a record can
  never contribute to its own feature value.
* `jurisdiction_recent_median_days_60d` — the realised turnaround of permits
  **already closed** in the 60 days before `start_date`. Those durations are
  fully observed at time *t*.
* `jurisdiction_open_permits` — permits filed before *t* that have not yet
  closed as of *t*. Whether a previous permit is still open is knowable on day
  *t*; only its eventual close date is unknown, and that is not used.
* Tracking the Sun context — as-of joins over utility records installed
  strictly before `start_date`.

Five tests enforce this rather than trusting the reasoning:

| Test | What it proves |
|---|---|
| `test_contractor_history_excludes_the_current_row` | a contractor's first job has `NaN` history, and every later row's prior mean equals the mean of strictly earlier targets |
| `test_jurisdiction_recent_stats_use_only_already_finished_jobs` | the 60-day window is recomputed by brute force and must match exactly |
| `test_open_permit_count_is_observable_at_start_date` | the open-permit count is recomputed by brute force |
| `test_jurisdiction_application_counts_are_strictly_prior` | same-day siblings are excluded from the trailing-window counts — these registers are day-resolution and give no intra-day ordering, so counting them would be a look-ahead |
| `test_shuffling_the_target_destroys_signal` | with a permuted target the model must score R² < 0.05 — if any feature encoded the row's own target it would not |

**This caught a real bug.** `_contractor_features` originally used
`groupby(...).apply(...)`, which returns rows *grouped by key* rather than in
the caller's order; writing that back positionally with `np.where` leaked a
contractor's own target into its first record. It inflated `application_to_issue`
CV MAE from a true ~6.1 days to an apparent ~3.4 days. The fix is
`transform`, which preserves row order. Every number in this repo was measured
**after** that fix.

## Deliberately excluded features

`start_year` and `days_since_series_start` are **not** features. They let the
booster memorise each historical period's level, which inflates random-fold CV
but collapses on a chronological holdout because the value is out of range for
future permits. Measured on `application_to_issue`: stock XGBoost went from
test R² **−0.10** with them to **+0.17** without.

## Evaluation protocol

* **Tuning**: 5-fold `KFold(shuffle=True, random_state=20250201)` over the
  training window only. The Optuna objective is CV MAE. The holdout is never
  touched during the search.
* **Holdout**: chronological, and **split inside each jurisdiction** (each city
  contributes its own most recent 20%). A single global cut-off would change the
  *mix of cities* between train and test as well as the time period — the
  registers' coverage windows differ by up to a decade — confounding "does this
  generalise forward in time?" with "does this generalise to a different city?".
  The global split is available via `split_strategy="global"`.
* **Baselines**, all scored on the identical folds:
  1. `global_median` — training median for everything
  2. `group_median` — median of the (jurisdiction, review_track) cell, i.e. the
     rule a permit desk already applies
  3. `ridge` — Ridge on median-imputed numerics + one-hot categoricals
  4. `xgb_default` — `XGBRegressor()` with stock hyper-parameters. **This is the
     baseline the "improvement from hyper-parameter search" figure is measured
     against.**

## Hyper-parameter search

Optuna `TPESampler(multivariate=True, seed=20250201, n_startup_trials=25)`,
200 trials, minimising 5-fold CV MAE. The space covers `objective`
(squared error / absolute error / pseudo-Huber), a `log1p` target transform,
tree shape (`max_depth`, `grow_policy`, `max_leaves`, `min_child_weight`),
sampling (`subsample`, `colsample_bytree`, `colsample_bylevel`), regularisation
(`gamma`, `reg_alpha`, `reg_lambda`), `n_estimators`, `learning_rate`,
`max_bin` and `max_cat_to_onehot`.

Every trial is one MLflow run holding its sampled hyper-parameters, aggregate
CV metrics and per-fold metrics.

## Serving

The API cannot recompute contractor history or jurisdiction-queue state from a
request body, and shipping the corpus into the service would blow the latency
budget. Instead `msa/features/store.py` snapshots the last observed value of
every context feature per contractor / jurisdiction / ZIP / state at the end of
training. The service is then a pure function of *(request, snapshot)*: no disk
reads, no network calls, one `XGBRegressor.predict` on a single row.

Staleness is explicit — `/metadata` reports `feature_store_as_of`. Refreshing
means re-running the pipeline.

### Known train/serve gaps

Two features cannot be reconstructed exactly from a request plus a snapshot, and
both are handled by leaving the value **missing** rather than substituting
something that would mean a different thing than it did in training:

| Feature | Training | Serving |
|---|---|---|
| `contractor_days_since_last_app` | days since that contractor's previous filing | missing — needs filing history relative to the request date |
| `jurisdiction_*` queue state | recomputed per row | last observed value for that jurisdiction, dated by `feature_store_as_of` |

Everything else — calendar features, size/cost transforms, the zoning-review
flag — is derived by the same code path in both places.
`tests/test_serve.py::test_request_frame_matches_training_columns_and_levels`
and `::test_zoning_review_flag_is_derived_the_same_way_as_in_training` pin that.
