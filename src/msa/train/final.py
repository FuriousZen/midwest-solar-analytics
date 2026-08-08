"""Fit the final model for a segment, score it, and register it with MLflow.

Consumes ``reports/<segment>/best_params.json`` from the Optuna study (falling
back to stock hyper-parameters if the study has not been run) and produces:

* ``models/<segment>/model.json``     -- the booster, in XGBoost's own format
* ``models/<segment>/metadata.json``  -- feature order, categorical levels,
  training window, metrics; everything the FastAPI service needs to rebuild an
  identical feature frame at request time
* an MLflow run tagged ``stage=final`` holding the same metrics plus the logged
  model artifact with an inferred signature

Run::

    python -m msa.train.final --segment permit_to_completion
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature

from msa.config import (
    DEFAULT_SEGMENT,
    MODELS_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
    SEGMENTS,
    max_timeline_days,
)
from msa.features.build import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES
from msa.features.store import build_store, save_store
from msa.train.baseline import N_SPLITS, default_xgb, make_cv, _oof_sklearn
from msa.train.data import Dataset, load_dataset, load_raw
from msa.train.metrics import improvement_pct, regression_metrics
from msa.train.tracking import init_mlflow
from msa.train.tune import cv_evaluate, make_model

LOG = logging.getLogger(__name__)


def model_dir(segment: str) -> Path:
    d = MODELS_DIR / segment
    d.mkdir(parents=True, exist_ok=True)
    return d


def report_dir(segment: str) -> Path:
    d = REPORTS_DIR / segment
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_best_params(segment: str) -> tuple[dict, bool, int | None]:
    """Return ``(xgb_params, log1p_target, best_trial_number)``."""
    path = report_dir(segment) / "best_params.json"
    if not path.exists():
        LOG.warning("%s not found -- falling back to stock hyper-parameters", path)
        return {}, False, None
    params = json.loads(path.read_text())
    log_target = bool(params.pop("log1p_target", False))
    study = report_dir(segment) / "optuna_study.json"
    trial = None
    if study.exists():
        trial = json.loads(study.read_text()).get("best_trial_number")
    return params, log_target, trial


#: Interval endpoints published alongside the point prediction.
QUANTILES = (0.1, 0.5, 0.9)


def fit_final(ds: Dataset, params: dict, log_target: bool):
    model = make_model(params)
    y = np.log1p(ds.y_train) if log_target else ds.y_train
    model.fit(ds.X_train, y, verbose=False)
    return model


def fit_quantiles(ds: Dataset, params: dict, log_target: bool):
    """Companion multi-quantile model giving an 80% prediction interval.

    A point estimate is close to useless on a target whose IQR spans months, so
    the service publishes p10/p50/p90 as well. XGBoost 3.x fits all three in one
    booster via ``objective="reg:quantileerror"`` with a vector
    ``quantile_alpha`` (verified empirically against 3.4.0 -- it is accepted
    through ``**kwargs`` and is absent from ``get_params()``).

    The tuned tree-shape / regularisation hyper-parameters are reused; only the
    objective changes. Quantiles are invariant under the monotone ``expm1``
    back-transform, so a log-target model needs no correction.
    """
    from xgboost import XGBRegressor

    shape = {
        k: v
        for k, v in params.items()
        if k not in ("objective", "huber_slope", "eval_metric")
    }
    model = XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=np.array(QUANTILES),
        enable_categorical=True,
        tree_method="hist",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=0,
        **shape,
    )
    y = np.log1p(ds.y_train) if log_target else ds.y_train
    model.fit(ds.X_train, y, verbose=False)
    return model


def predict_quantiles(model, X: pd.DataFrame, log_target: bool) -> np.ndarray:
    """Return an ``(n, len(QUANTILES))`` array in days, sorted per row."""
    pred = np.atleast_2d(model.predict(X))
    if pred.shape[0] != len(X):
        pred = pred.T
    if log_target:
        pred = np.expm1(pred)
    pred = np.clip(pred, 0.0, None)
    # crossing quantiles are possible with independent pinball losses; enforce
    # monotonicity so the published interval is always well formed
    return np.sort(pred, axis=1)


#: Fraction of the (chronologically ordered) training window reserved for
#: conformal calibration. Never overlaps the holdout.
CALIBRATION_FRACTION = 0.15


def calibrate_conformal(ds: Dataset, params: dict, log_target: bool) -> float:
    """Split-conformal (CQR) widening for the quantile interval.

    Pinball-loss quantiles are not calibrated out of the box -- on this data the
    nominal 80% interval empirically covered ~64% of the holdout. Conformalised
    Quantile Regression (Romano, Patterson & Candes, 2019) fixes that with a
    single additive correction estimated on held-out-from-fit data:

        s_i = max(lo_i - y_i,  y_i - hi_i)         # conformity score
        delta = the ceil((n+1)(1-alpha))/n empirical quantile of s

    The interval [lo - delta, hi + delta] then has finite-sample coverage of at
    least 1 - alpha, under exchangeability.

    The calibration slice is the **most recent 15% of the training window** --
    chronological, so it never touches the holdout and never lets future
    information into the fit.
    """
    n = len(ds.X_train)
    cut = int(n * (1.0 - CALIBRATION_FRACTION))
    if cut < 50 or n - cut < 50:
        LOG.warning("training window too small to conformalise; delta = 0")
        return 0.0

    fit_ds = Dataset(
        X_train=ds.X_train.iloc[:cut],
        y_train=ds.y_train.iloc[:cut],
        X_test=ds.X_train.iloc[cut:],
        y_test=ds.y_train.iloc[cut:],
        meta_train=ds.meta_train.iloc[:cut],
        meta_test=ds.meta_train.iloc[cut:],
        split_date=ds.split_date,
        categories=ds.categories,
        segment=ds.segment,
    )
    model = fit_quantiles(fit_ds, params, log_target)
    q = predict_quantiles(model, fit_ds.X_test, log_target)
    y = fit_ds.y_test.to_numpy()
    scores = np.maximum(q[:, 0] - y, y - q[:, 2])
    alpha = 1.0 - (QUANTILES[2] - QUANTILES[0])
    m = len(scores)
    level = min(np.ceil((m + 1) * (1 - alpha)) / m, 1.0)
    delta = float(np.quantile(scores, level, method="higher"))
    LOG.info(
        "conformal calibration on %d held-out training rows: delta = %+.1f days",
        m,
        delta,
    )
    return delta


def predict(model, X: pd.DataFrame, log_target: bool) -> np.ndarray:
    pred = model.predict(X)
    if log_target:
        pred = np.expm1(pred)
    return np.clip(pred, 0.0, None)


def build_metadata(ds: Dataset, params: dict, log_target: bool, metrics: dict) -> dict:
    start, end = SEGMENTS[ds.segment]
    return {
        "segment": ds.segment,
        "target": "timeline_days",
        "target_definition": f"({end} - {start}).days",
        "horizon_days": max_timeline_days(ds.segment),
        "features": list(FEATURES),
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "categories": ds.categories,
        "log1p_target": log_target,
        "xgb_params": params,
        "n_train": ds.n_train,
        "n_test": ds.n_test,
        "train_start": str(ds.meta_train["start_date"].min().date()),
        "train_end": str(ds.meta_train["start_date"].max().date()),
        "test_start": str(ds.meta_test["start_date"].min().date()),
        "test_end": str(ds.meta_test["start_date"].max().date()),
        "jurisdictions": sorted(ds.meta_train["jurisdiction"].unique().tolist()),
        "seed": RANDOM_SEED,
        "metrics": metrics,
        "train_median_days": float(ds.y_train.median()),
        "quantiles": list(QUANTILES),
        "conformal_delta_days": float(metrics.get("conformal_delta_days", 0.0)),
        "interval_empirical_coverage_test": metrics.get(
            "interval_empirical_coverage_test"
        ),
    }


def run(segment: str, log_to_mlflow: bool = True) -> dict:
    ds = load_dataset(segment=segment)
    params, log_target, trial_no = load_best_params(segment)
    cv = make_cv()

    tuned_cv, tuned_folds, tuned_oof = cv_evaluate(
        params, ds.X_train, ds.y_train, cv, log_target
    )
    base_oof = _oof_sklearn(default_xgb, ds.X_train, ds.y_train, make_cv())
    base_cv = regression_metrics(ds.y_train, base_oof, prefix="cv_")

    model = fit_final(ds, params, log_target)
    test_pred = predict(model, ds.X_test, log_target)
    test = regression_metrics(ds.y_test, test_pred, prefix="test_")

    conformal_delta = calibrate_conformal(ds, params, log_target)
    qmodel = fit_quantiles(ds, params, log_target)
    q_test = predict_quantiles(qmodel, ds.X_test, log_target)
    raw_lo, mid, raw_hi = q_test[:, 0], q_test[:, 1], q_test[:, 2]
    lo = np.clip(raw_lo - conformal_delta, 0.0, None)
    hi = raw_hi + conformal_delta
    y_true = ds.y_test.to_numpy()
    interval = {
        "interval_nominal_coverage": QUANTILES[2] - QUANTILES[0],
        "conformal_delta_days": float(conformal_delta),
        "interval_empirical_coverage_test": float(
            ((y_true >= lo) & (y_true <= hi)).mean()
        ),
        "interval_uncalibrated_coverage_test": float(
            ((y_true >= raw_lo) & (y_true <= raw_hi)).mean()
        ),
        "interval_median_width_days": float(np.median(hi - lo)),
        "interval_mean_width_days": float(np.mean(hi - lo)),
        "quantile_median_test_mae": float(np.mean(np.abs(ds.y_test - mid))),
    }

    # a naive reference on the same holdout
    naive = np.full(ds.n_test, float(ds.y_train.median()))
    naive_test = regression_metrics(ds.y_test, naive, prefix="naive_test_")

    metrics = {
        **tuned_cv,
        **test,
        **naive_test,
        **interval,
        "baseline_xgb_default_cv_mae": base_cv["cv_mae"],
        "baseline_xgb_default_cv_r2": base_cv["cv_r2"],
        "baseline_xgb_default_cv_rmse": base_cv["cv_rmse"],
        "improvement_vs_default_cv_mae_pct": improvement_pct(
            base_cv["cv_mae"], tuned_cv["cv_mae"]
        ),
        "improvement_vs_default_cv_rmse_pct": improvement_pct(
            base_cv["cv_rmse"], tuned_cv["cv_rmse"]
        ),
        "improvement_vs_naive_test_mae_pct": improvement_pct(
            naive_test["naive_test_mae"], test["test_mae"]
        ),
        "best_trial_number": trial_no,
    }

    md = model_dir(segment)
    model.save_model(str(md / "model.json"))
    qmodel.save_model(str(md / "model_quantiles.json"))

    # snapshot the context features the API cannot recompute from a request
    from msa.features.build import clean

    canonical = clean(load_raw(), segment=segment)
    store = build_store(ds.X_train, ds.meta_train, canonical)
    save_store(store, md / "feature_store.json")
    meta = build_metadata(ds, params, log_target, metrics)
    meta["feature_store_as_of"] = store["as_of"]
    (md / "metadata.json").write_text(json.dumps(meta, indent=2))

    rd = report_dir(segment)
    (rd / "final_metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame(
        {
            "record_id": ds.meta_test["record_id"],
            "jurisdiction": ds.meta_test["jurisdiction"],
            "start_date": ds.meta_test["start_date"],
            "y_true": ds.y_test,
            "y_pred": test_pred,
            "y_p10": lo,
            "y_p50": mid,
            "y_p90": hi,
        }
    ).to_csv(rd / "test_predictions.csv", index=False)
    pd.DataFrame({"y_true": ds.y_train, "y_oof": tuned_oof}).to_csv(
        rd / "oof_predictions.csv", index=False
    )

    if log_to_mlflow:
        with mlflow.start_run(run_name=f"final-{segment}"):
            mlflow.set_tags(
                {
                    "stage": "final",
                    "segment": segment,
                    "model_family": "xgboost",
                    "cv_folds": str(N_SPLITS),
                }
            )
            mlflow.log_params(
                {
                    **params,
                    "log1p_target": log_target,
                    "segment": segment,
                    "n_train": ds.n_train,
                    "n_test": ds.n_test,
                    "cv_splits": N_SPLITS,
                    "seed": RANDOM_SEED,
                }
            )
            mlflow.log_metrics({k: v for k, v in metrics.items() if v is not None})
            example = ds.X_test.head(3)
            mlflow.xgboost.log_model(
                model,
                name=f"model-{segment}",
                signature=infer_signature(example, predict(model, example, log_target)),
                input_example=example,
            )
            mlflow.log_artifact(str(md / "metadata.json"))
            mlflow.log_artifact(str(rd / "final_metrics.json"))

    LOG.info(
        "[%s] 80%% interval covers %.1f%% of the holdout, median width %.0f days",
        segment,
        interval["interval_empirical_coverage_test"] * 100,
        interval["interval_median_width_days"],
    )
    LOG.info(
        "[%s] tuned cv_mae=%.3f cv_r2=%.4f | test_mae=%.3f test_r2=%.4f | "
        "vs stock XGBoost: %.1f%% lower CV MAE",
        segment,
        tuned_cv["cv_mae"],
        tuned_cv["cv_r2"],
        test["test_mae"],
        test["test_r2"],
        metrics["improvement_vs_default_cv_mae_pct"],
    )
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment", default=DEFAULT_SEGMENT, choices=list(SEGMENTS))
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if not args.no_mlflow:
        init_mlflow()
    metrics = run(args.segment, log_to_mlflow=not args.no_mlflow)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
