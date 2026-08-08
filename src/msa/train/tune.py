"""200-trial Optuna hyper-parameter search for the XGBoost timeline regressor.

Protocol
--------
* Search algorithm : Optuna ``TPESampler`` (seeded, ``multivariate=True``).
* Objective        : mean absolute error, averaged over a 5-fold
                     ``KFold(shuffle=True)`` cross-validation of the *training*
                     window only. The chronological holdout is never touched
                     during the search.
* Baseline         : the ``xgb_default`` row from ``msa.train.baseline`` -- the
                     same 5-fold CV, same folds, stock XGBoost hyper-parameters.
* Tracking         : one MLflow run per trial (nested inside a parent run),
                     recording every sampled hyper-parameter and the fold-level
                     and aggregate metrics.

Run::

    python -m msa.train.tune --trials 200
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import mlflow
import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

from msa.config import DEFAULT_SEGMENT, RANDOM_SEED, REPORTS_DIR, SEGMENTS
from msa.train.baseline import N_SPLITS, make_cv
from msa.train.data import Dataset, load_dataset
from msa.train.metrics import improvement_pct, regression_metrics
from msa.train.tracking import init_mlflow

LOG = logging.getLogger(__name__)

def report_dir(segment: str):
    d = REPORTS_DIR / segment
    d.mkdir(parents=True, exist_ok=True)
    return d

#: Objectives worth trying for a skewed, non-negative duration target.
OBJECTIVES = ("reg:squarederror", "reg:absoluteerror", "reg:pseudohubererror")


def suggest_params(trial: optuna.Trial) -> dict:
    """Sample one XGBoost configuration.

    Every argument name below is a real ``XGBRegressor`` constructor parameter
    (verified against ``XGBRegressor().get_params()`` for xgboost 3.4.0).
    """
    grow_policy = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    params: dict = {
        "objective": trial.suggest_categorical("objective", list(OBJECTIVES)),
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 5e-3, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 12),
        "min_child_weight": trial.suggest_float(
            "min_child_weight", 0.5, 60.0, log=True
        ),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.4, 1.0),
        "gamma": trial.suggest_float("gamma", 1e-8, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 20.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True),
        "max_bin": trial.suggest_categorical("max_bin", [128, 256, 512]),
        "max_cat_to_onehot": trial.suggest_int("max_cat_to_onehot", 1, 12),
        "grow_policy": grow_policy,
    }
    if grow_policy == "lossguide":
        params["max_leaves"] = trial.suggest_int("max_leaves", 4, 256, log=True)
    if params["objective"] == "reg:pseudohubererror":
        params["huber_slope"] = trial.suggest_float("huber_slope", 0.5, 20.0, log=True)
    return params


def make_model(params: dict) -> XGBRegressor:
    return XGBRegressor(
        enable_categorical=True,
        tree_method="hist",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=0,
        **params,
    )


def cv_evaluate(
    params: dict,
    X: pd.DataFrame,
    y: pd.Series,
    cv: KFold,
    log_target: bool,
) -> tuple[dict[str, float], list[dict[str, float]], np.ndarray]:
    """Return (aggregate metrics, per-fold metrics, out-of-fold predictions)."""
    oof = np.empty(len(y), dtype=float)
    fold_metrics: list[dict[str, float]] = []
    for fold, (tr, va) in enumerate(cv.split(X)):
        model = make_model(params)
        y_tr = np.log1p(y.iloc[tr]) if log_target else y.iloc[tr]
        model.fit(X.iloc[tr], y_tr, verbose=False)
        pred = model.predict(X.iloc[va])
        if log_target:
            pred = np.expm1(pred)
        pred = np.clip(pred, 0.0, None)
        oof[va] = pred
        fm = regression_metrics(y.iloc[va], pred)
        fm["fold"] = fold
        fold_metrics.append(fm)
    agg = regression_metrics(y, oof, prefix="cv_")
    agg["cv_mae_fold_std"] = float(np.std([m["mae"] for m in fold_metrics]))
    return agg, fold_metrics, oof


def run_study(
    ds: Dataset,
    n_trials: int = 200,
    log_to_mlflow: bool = True,
    study_name: str = "xgb-timeline-tpe",
) -> optuna.Study:
    X, y = ds.X_train, ds.y_train
    cv = make_cv()

    sampler = TPESampler(seed=RANDOM_SEED, multivariate=True, n_startup_trials=25)
    study = optuna.create_study(
        direction="minimize", sampler=sampler, study_name=study_name
    )

    def objective(trial: optuna.Trial) -> float:
        log_target = trial.suggest_categorical("log1p_target", [False, True])
        params = suggest_params(trial)
        t0 = time.perf_counter()
        agg, folds, _ = cv_evaluate(params, X, y, cv, log_target)
        elapsed = time.perf_counter() - t0

        if log_to_mlflow:
            with mlflow.start_run(run_name=f"optuna-trial-{trial.number:03d}", nested=True):
                mlflow.set_tags(
                    {
                        "stage": "optuna",
                        "study": study_name,
                        "segment": ds.segment,
                        "trial_number": str(trial.number),
                        "model_family": "xgboost",
                    }
                )
                mlflow.log_params(
                    {**params, "log1p_target": log_target, "cv_splits": N_SPLITS}
                )
                mlflow.log_metrics({**agg, "fit_seconds": elapsed})
                for fm in folds:
                    mlflow.log_metric("fold_mae", fm["mae"], step=int(fm["fold"]))
                    mlflow.log_metric("fold_r2", fm["r2"], step=int(fm["fold"]))
        trial.set_user_attr("cv_r2", agg["cv_r2"])
        trial.set_user_attr("cv_rmse", agg["cv_rmse"])
        trial.set_user_attr("seconds", elapsed)
        return agg["cv_mae"]

    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
    return study


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--segment", default=DEFAULT_SEGMENT, choices=list(SEGMENTS))
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()
    rd = report_dir(args.segment)
    study_path = rd / "optuna_study.json"
    best_params_path = rd / "best_params.json"
    trials_csv = rd / "optuna_trials.csv"

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    ds = load_dataset(segment=args.segment)
    parent = None
    if not args.no_mlflow:
        init_mlflow()
        parent = mlflow.start_run(
            run_name=f"optuna-study-{args.segment}-{args.trials}-trials"
        )
        mlflow.set_tags(
            {
                "stage": "optuna-parent",
                "segment": args.segment,
                "trials": str(args.trials),
            }
        )

    try:
        t0 = time.perf_counter()
        study = run_study(
            ds,
            n_trials=args.trials,
            log_to_mlflow=not args.no_mlflow,
            study_name=f"xgb-{args.segment}-tpe",
        )
        wall = time.perf_counter() - t0

        # baseline reference: identical folds, stock hyper-parameters
        from msa.train.baseline import default_xgb, _oof_sklearn

        base_oof = _oof_sklearn(default_xgb, ds.X_train, ds.y_train, make_cv())
        base = regression_metrics(ds.y_train, base_oof, prefix="cv_")

        best = study.best_trial
        payload = {
            "segment": args.segment,
            "study_name": study.study_name,
            "sampler": "TPESampler(multivariate=True, seed=%d)" % RANDOM_SEED,
            "n_trials": len(study.trials),
            "cv_splits": N_SPLITS,
            "cv_type": "KFold(shuffle=True, random_state=%d)" % RANDOM_SEED,
            "wall_seconds": wall,
            "n_train": ds.n_train,
            "best_trial_number": best.number,
            "best_cv_mae": best.value,
            "best_cv_r2": best.user_attrs.get("cv_r2"),
            "best_cv_rmse": best.user_attrs.get("cv_rmse"),
            "best_params": best.params,
            "baseline_xgb_default": base,
            "improvement_vs_default_mae_pct": improvement_pct(
                base["cv_mae"], best.value
            ),
            "improvement_vs_default_rmse_pct": improvement_pct(
                base["cv_rmse"], best.user_attrs.get("cv_rmse", float("nan"))
            ),
        }
        study_path.write_text(json.dumps(payload, indent=2, default=str))
        best_params_path.write_text(json.dumps(best.params, indent=2))
        study.trials_dataframe().to_csv(trials_csv, index=False)

        if not args.no_mlflow:
            mlflow.log_params(
                {
                    "n_trials": args.trials,
                    "cv_splits": N_SPLITS,
                    "segment": args.segment,
                }
            )
            mlflow.log_metrics(
                {
                    "best_cv_mae": float(best.value),
                    "best_cv_r2": float(best.user_attrs.get("cv_r2", np.nan)),
                    "baseline_cv_mae": base["cv_mae"],
                    "improvement_vs_default_mae_pct": payload[
                        "improvement_vs_default_mae_pct"
                    ],
                    "wall_seconds": wall,
                }
            )
            mlflow.log_artifact(str(study_path))
            mlflow.log_artifact(str(trials_csv))
        print(json.dumps(payload, indent=2, default=str))
    finally:
        if parent is not None:
            mlflow.end_run()


if __name__ == "__main__":  # pragma: no cover
    main()
