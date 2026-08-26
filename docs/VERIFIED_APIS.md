# Verified library APIs

Every third-party API used in this repo was checked against the installed
version before being written, either by fetching the official documentation
or by introspecting the installed package (`inspect.signature`,
`get_params()`, docstrings). Nothing here is from memory.

Reproduce the introspection checks with:

```bash
.venv/bin/python -m scripts.verify_apis
```

| Library | Version | What was checked | Result |
|---|---|---|---|
| xgboost | 3.4.0 | `XGBRegressor.__init__` accepts `**kwargs`; the real parameter list comes from `XGBRegressor().get_params()` | Confirmed: `early_stopping_rounds`, `enable_categorical`, `tree_method`, `device`, `max_cat_to_onehot`, `grow_policy`, `max_leaves`, `max_bin`, `callbacks` are all constructor parameters |
| xgboost | 3.4.0 | `XGBRegressor.fit` signature | `fit(X, y, *, sample_weight, base_margin, eval_set, verbose, xgb_model, sample_weight_eval_set, base_margin_eval_set, feature_weights)`. `early_stopping_rounds` is not a `fit` argument. It moved to the constructor in 1.6.0 |
| xgboost | 3.4.0 | native categorical support | `enable_categorical=True` with pandas `category` dtype; `tree_method="hist"` |
| optuna | 4.9.0 | `create_study`, `Study.optimize`, `Trial.suggest_float/int`, `TPESampler.__init__` | `create_study` is keyword-only; `suggest_float(name, low, high, *, step, log)`; `TPESampler(..., seed, multivariate, n_startup_trials)`. `multivariate=True` emits an `ExperimentalWarning`, accepted deliberately |
| mlflow | 3.15.1 | `mlflow.xgboost.log_model` | Signature has both `artifact_path` (deprecated) and `name`; this repo uses `name=`. Also uses `signature=`, `input_example=` |
| mlflow | 3.15.1 | `mlflow.xgboost.autolog` compatibility | Docs state autologging is verified for `2.1.1 <= xgboost <= 3.3.0`. We run 3.4.0, so autolog is deliberately not used; every param and metric is logged explicitly |
| mlflow | 3.15.1 | tracking backend | The legacy `./mlruns` file store raises `MlflowException` ("in maintenance mode") on 3.x. This repo uses the recommended `sqlite:///mlflow.db` backend |
| mlflow | 3.15.1 | `mlflow.search_runs` | `(experiment_ids, filter_string, run_view_type, max_results, order_by, output_format, search_all_experiments, experiment_names)` |
| shap | 0.52.0 | `TreeExplainer.__init__`, `shap.plots.*` | `TreeExplainer(model, data=None, model_output='raw', feature_perturbation='auto', ...)`; available plots include `beeswarm`, `bar`, `waterfall`, `scatter`, `heatmap` |
| scikit-learn | 1.9.0 | `KFold.__init__` | `(n_splits=5, *, shuffle=False, random_state=None)` |
| fastapi / pydantic | 0.141.1 / 2.13.4 | v2 style: `BaseModel`, `Field`, `field_validator`, `model_dump()` | Confirmed in use |
| pandas | 2.3.3 | `merge_asof` join-key dtype rules | `by=` keys must have *identical* dtypes on both sides: mixing `string[python]` with `object` raises `MergeError`. Both sides are cast to `string` explicitly |

## Data-source API notes (also verified live)

| Source | Note |
|---|---|
| Socrata SODA 2.1 | `like` is case-sensitive; queries wrap the column in `upper(...)`. A literal `%` must be passed to `--data-urlencode` (double-encoding `%25` silently matches nothing and returns `count = 0`) |
| Socrata | `$order` is required for stable offset pagination |
| ArcGIS REST | `esriFieldTypeDate` returns epoch milliseconds; `esriFieldTypeDateOnly` returns an ISO date string. Detroit's BSEED layer uses DateOnly, Minneapolis uses Date. `msa.ingest.arcgis.parse_esri_date` handles both, and `tests/test_ingest.py` locks that behaviour in |
| ArcGIS REST | truncation is signalled by `exceededTransferLimit`; page with `resultOffset`/`resultRecordCount` and an explicit `orderByFields` |
| OEDI S3 (`oedi-data-lake`) | anonymous HTTPS listing works via `?list-type=2&prefix=...`; Tracking the Sun is Hive-partitioned `<edition>/state=XX/` |

## Runtime assumptions the service depends on

Each of these was confirmed against the pinned version (and, where noted,
against the official documentation) rather than recalled. `python -m
scripts.verify_apis` asserts the mechanical ones on every run.

| library | assumption | how it was verified | why it matters |
|---|---|---|---|
| `httpx` 0.28.1 | `ASGITransport` is async-only: it subclasses `AsyncBaseTransport` and has no `__enter__`, so it must be driven by `AsyncClient` | introspection + [transports docs](https://www.python-httpx.org/advanced/transports/) | `msa.bench.latency` and `scripts.demo_predict` drive the app in-process through it, so both are written as async code |
| `xgboost` 3.4.0 | all prediction functions, including SHAP contributions, are thread-safe for `gbtree`/`dart` boosters as long as nothing mutates the model concurrently | [prediction docs](https://xgboost.readthedocs.io/en/stable/prediction.html) | the service shares one booster across the whole worker thread-pool with no lock. Nothing calls `set_params()` at request time, which is the documented way to break the guarantee |
| `shap` 0.52.0 | `TreeExplainer.shap_values` on an XGBoost model delegates to `booster.predict(..., pred_contribs=True)` and does not mutate the explainer on that path | reading the installed `shap/explainers/_tree.py` | concurrent `?explain=true` is therefore safe; it is still capped at `MSA_MAX_CONCURRENT_EXPLAIN` to bound memory, not for correctness |
| `pydantic` 2.13.4 | `allow_inf_nan` is available both on `ConfigDict` and per-`Field`; without it a JSON `Infinity` token passes a `ge=0` bound | [config docs](https://pydantic.dev/docs/validation/latest/api/pydantic/config/) + an executed check in `scripts/verify_apis.py` | `Infinity` flowed through `log1p` and `cost_per_kw` into a non-finite prediction |
| `fastapi` 0.141.1 | `lifespan=` async context manager is the supported startup hook; `@app.exception_handler(Exception)` installs Starlette's `ServerErrorMiddleware` handler, which runs *outside* user middleware | [lifespan docs](https://fastapi.tiangolo.com/advanced/events/), [error docs](https://fastapi.tiangolo.com/tutorial/handling-errors/) | the request id is therefore read from `scope["state"]` in the 500 handler, because the context variable has already been reset by then |
| `uvicorn` 0.52.1 | an exception raised inside the `lifespan` scope is interpreted as *"the app does not implement lifespan"*: uvicorn logs `ASGI 'lifespan' protocol appears unsupported` and serves anyway. The correct refusal is to send `{"type": "lifespan.startup.failed"}` | run against a real uvicorn process; `exit 3` and no socket bound | this is the difference between "refuses to start without a model" and "starts with no model and 500s every request" |
| `mlflow` 3.15.1 | `mlflow.xgboost.log_model` produces a model that cannot be served: MLflow carries DataFrames as JSON, JSON has no categorical dtype, and the booster rejects the resulting `object` columns. MLflow only *warns* about this | reproduced with a real `final` run; the model is logged through a `pyfunc` wrapper instead and `tests/test_train_artifacts.py` asserts it can be served | without the wrapper the registry would hold a model that loads but cannot predict |
| `optuna` 4.9.0 | `study.tell(trial_number, state=TrialState.FAIL)` retires a stale `RUNNING` trial left by a killed process | executed against `optuna.db` | a killed search must be resumable without a trial stuck in `RUNNING` |
