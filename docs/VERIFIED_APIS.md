# Verified library APIs

Every third-party API used in this repo was checked against the **installed
version** before being written, either by fetching the official documentation
or by introspecting the installed package (`inspect.signature`,
`get_params()`, docstrings). Nothing here is from memory.

Reproduce the introspection checks with:

```bash
.venv/bin/python -m scripts.verify_apis
```

| Library | Version | What was checked | Result |
|---|---|---|---|
| xgboost | 3.4.0 | `XGBRegressor.__init__` accepts `**kwargs`; the real parameter list comes from `XGBRegressor().get_params()` | Confirmed: `early_stopping_rounds`, `enable_categorical`, `tree_method`, `device`, `max_cat_to_onehot`, `grow_policy`, `max_leaves`, `max_bin`, `callbacks` are all constructor parameters |
| xgboost | 3.4.0 | `XGBRegressor.fit` signature | `fit(X, y, *, sample_weight, base_margin, eval_set, verbose, xgb_model, sample_weight_eval_set, base_margin_eval_set, feature_weights)`. **`early_stopping_rounds` is NOT a `fit` argument** — it moved to the constructor in 1.6.0 |
| xgboost | 3.4.0 | native categorical support | `enable_categorical=True` with pandas `category` dtype; `tree_method="hist"` |
| optuna | 4.9.0 | `create_study`, `Study.optimize`, `Trial.suggest_float/int`, `TPESampler.__init__` | `create_study` is keyword-only; `suggest_float(name, low, high, *, step, log)`; `TPESampler(..., seed, multivariate, n_startup_trials)`. `multivariate=True` emits an `ExperimentalWarning` — accepted deliberately |
| mlflow | 3.15.1 | `mlflow.xgboost.log_model` | Signature has **both** `artifact_path` (deprecated) and `name`; this repo uses `name=`. Also uses `signature=`, `input_example=` |
| mlflow | 3.15.1 | `mlflow.xgboost.autolog` compatibility | Docs state autologging is verified for `2.1.1 <= xgboost <= 3.3.0`. We run **3.4.0**, so autolog is deliberately *not* used — every param and metric is logged explicitly |
| mlflow | 3.15.1 | tracking backend | The legacy `./mlruns` file store raises `MlflowException` ("in maintenance mode") on 3.x. This repo uses the recommended `sqlite:///mlflow.db` backend |
| mlflow | 3.15.1 | `mlflow.search_runs` | `(experiment_ids, filter_string, run_view_type, max_results, order_by, output_format, search_all_experiments, experiment_names)` |
| shap | 0.52.0 | `TreeExplainer.__init__`, `shap.plots.*` | `TreeExplainer(model, data=None, model_output='raw', feature_perturbation='auto', ...)`; available plots include `beeswarm`, `bar`, `waterfall`, `scatter`, `heatmap` |
| scikit-learn | 1.9.0 | `KFold.__init__` | `(n_splits=5, *, shuffle=False, random_state=None)` |
| fastapi / pydantic | 0.141.1 / 2.13.4 | v2 style: `BaseModel`, `Field`, `field_validator`, `model_dump()` | Confirmed in use |
| pandas | 2.3.3 | `merge_asof` join-key dtype rules | `by=` keys must have *identical* dtypes on both sides — mixing `string[python]` with `object` raises `MergeError`. Both sides are cast to `string` explicitly |

## Data-source API notes (also verified live)

| Source | Note |
|---|---|
| Socrata SODA 2.1 | `like` is **case-sensitive**; queries wrap the column in `upper(...)`. A literal `%` must be passed to `--data-urlencode` (double-encoding `%25` silently matches nothing and returns `count = 0`) |
| Socrata | `$order` is required for stable offset pagination |
| ArcGIS REST | `esriFieldTypeDate` returns **epoch milliseconds**; `esriFieldTypeDateOnly` returns an **ISO date string**. Detroit's BSEED layer uses DateOnly, Minneapolis uses Date — `msa.ingest.arcgis.parse_esri_date` handles both, and `tests/test_ingest.py` locks that behaviour in |
| ArcGIS REST | truncation is signalled by `exceededTransferLimit`; page with `resultOffset`/`resultRecordCount` and an explicit `orderByFields` |
| OEDI S3 (`oedi-data-lake`) | anonymous HTTPS listing works via `?list-type=2&prefix=...`; Tracking the Sun is Hive-partitioned `<edition>/state=XX/` |
