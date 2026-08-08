"""Baseline models -- the reference points every later improvement is measured against.

Four baselines, all scored with the *same* 5-fold CV protocol used by the
Optuna study so the comparison is apples-to-apples:

``global_median``      predict the training median for everything.
``group_median``       predict the median of the (jurisdiction, review_track)
                       group -- i.e. the rule a permit desk already applies.
``ridge``              Ridge regression on median-imputed numerics + one-hot
                       categoricals.
``xgb_default``        ``XGBRegressor()`` with stock library hyper-parameters.
                       **This is the baseline for the "improvement from
                       hyper-parameter search" figure.**

Run::

    python -m msa.train.baseline
"""

from __future__ import annotations

import argparse
import json
import logging

import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from msa.config import DEFAULT_SEGMENT, RANDOM_SEED, REPORTS_DIR, SEGMENTS
from msa.features.build import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from msa.train.data import Dataset, load_dataset
from msa.train.metrics import regression_metrics
from msa.train.tracking import init_mlflow

LOG = logging.getLogger(__name__)

N_SPLITS = 5


def make_cv(n_splits: int = N_SPLITS) -> KFold:
    return KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)


# ---------------------------------------------------------------------------
# individual baselines -- each returns out-of-fold predictions
# ---------------------------------------------------------------------------


def _oof_global_median(X: pd.DataFrame, y: pd.Series, cv: KFold) -> np.ndarray:
    oof = np.empty(len(y))
    for tr, te in cv.split(X):
        oof[te] = float(np.median(y.iloc[tr]))
    return oof


def _oof_group_median(X: pd.DataFrame, y: pd.Series, cv: KFold) -> np.ndarray:
    keys = (
        X["jurisdiction"].astype(str) + "|" + X["review_track"].astype(str)
    ).to_numpy()
    oof = np.empty(len(y))
    for tr, te in cv.split(X):
        frame = pd.DataFrame({"k": keys[tr], "y": y.iloc[tr].to_numpy()})
        med = frame.groupby("k")["y"].median()
        fallback = float(np.median(y.iloc[tr]))
        oof[te] = pd.Series(keys[te]).map(med).fillna(fallback).to_numpy()
    return oof


def _ridge_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        (
                            "num",
                            Pipeline(
                                [
                                    ("imp", SimpleImputer(strategy="median")),
                                    ("sc", StandardScaler()),
                                ]
                            ),
                            list(NUMERIC_FEATURES),
                        ),
                        (
                            "cat",
                            OneHotEncoder(
                                handle_unknown="infrequent_if_exist",
                                min_frequency=25,
                                sparse_output=True,
                            ),
                            list(CATEGORICAL_FEATURES),
                        ),
                    ]
                ),
            ),
            ("model", Ridge(alpha=1.0, random_state=RANDOM_SEED)),
        ]
    )


def _oof_sklearn(model_factory, X: pd.DataFrame, y: pd.Series, cv: KFold) -> np.ndarray:
    oof = np.empty(len(y))
    for tr, te in cv.split(X):
        m = model_factory()
        Xtr = X.iloc[tr]
        if isinstance(m, Pipeline):
            Xtr = Xtr.copy()
            for c in CATEGORICAL_FEATURES:
                Xtr[c] = Xtr[c].astype(str)
            Xte = X.iloc[te].copy()
            for c in CATEGORICAL_FEATURES:
                Xte[c] = Xte[c].astype(str)
        else:
            Xte = X.iloc[te]
        m.fit(Xtr, y.iloc[tr])
        oof[te] = m.predict(Xte)
    return oof


def default_xgb(**overrides) -> XGBRegressor:
    """Stock XGBRegressor -- only the knobs needed to consume the data at all."""
    params = dict(
        enable_categorical=True,
        tree_method="hist",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    params.update(overrides)
    return XGBRegressor(**params)


# ---------------------------------------------------------------------------


def evaluate_baselines(ds: Dataset, log_to_mlflow: bool = True) -> dict:
    cv = make_cv()
    X, y = ds.X_train, ds.y_train

    specs = {
        "global_median": lambda: _oof_global_median(X, y, cv),
        "group_median": lambda: _oof_group_median(X, y, cv),
        "ridge": lambda: _oof_sklearn(_ridge_pipeline, X, y, cv),
        "xgb_default": lambda: _oof_sklearn(default_xgb, X, y, cv),
    }

    results: dict[str, dict] = {}
    for name, fn in specs.items():
        LOG.info("baseline: %s", name)
        oof = fn()
        cv_metrics = regression_metrics(y, oof, prefix="cv_")

        # refit on the whole training window and score the chronological holdout
        if name == "global_median":
            test_pred = np.full(ds.n_test, float(np.median(y)))
        elif name == "group_median":
            keys_tr = X["jurisdiction"].astype(str) + "|" + X["review_track"].astype(str)
            keys_te = (
                ds.X_test["jurisdiction"].astype(str)
                + "|"
                + ds.X_test["review_track"].astype(str)
            )
            med = pd.DataFrame({"k": keys_tr, "y": y}).groupby("k")["y"].median()
            test_pred = keys_te.map(med).fillna(float(np.median(y))).to_numpy()
        elif name == "ridge":
            m = _ridge_pipeline()
            Xtr = X.copy()
            Xte = ds.X_test.copy()
            for c in CATEGORICAL_FEATURES:
                Xtr[c] = Xtr[c].astype(str)
                Xte[c] = Xte[c].astype(str)
            m.fit(Xtr, y)
            test_pred = m.predict(Xte)
        else:
            m = default_xgb()
            m.fit(X, y)
            test_pred = m.predict(ds.X_test)

        test_metrics = regression_metrics(ds.y_test, test_pred, prefix="test_")
        results[name] = {**cv_metrics, **test_metrics}
        LOG.info(
            "  cv_mae=%.3f cv_r2=%.4f | test_mae=%.3f test_r2=%.4f",
            cv_metrics["cv_mae"],
            cv_metrics["cv_r2"],
            test_metrics["test_mae"],
            test_metrics["test_r2"],
        )

        if log_to_mlflow:
            with mlflow.start_run(run_name=f"baseline-{name}"):
                mlflow.set_tags(
                    {
                        "stage": "baseline",
                        "segment": ds.segment,
                        "model_family": name,
                        "cv_folds": str(N_SPLITS),
                    }
                )
                mlflow.log_params(
                    {
                        "baseline": name,
                        "segment": ds.segment,
                        "n_train": ds.n_train,
                        "n_test": ds.n_test,
                        "split_date": str(ds.split_date.date()),
                        "cv_splits": N_SPLITS,
                        "seed": RANDOM_SEED,
                    }
                )
                mlflow.log_metrics({**cv_metrics, **test_metrics})
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment", default=DEFAULT_SEGMENT, choices=list(SEGMENTS))
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if not args.no_mlflow:
        init_mlflow()
    ds = load_dataset(segment=args.segment)
    results = evaluate_baselines(ds, log_to_mlflow=not args.no_mlflow)
    payload = {
        "segment": args.segment,
        "n_train": ds.n_train,
        "n_test": ds.n_test,
        "split_date": str(ds.split_date.date()),
        "cv_splits": N_SPLITS,
        "seed": RANDOM_SEED,
        "results": results,
    }
    out_dir = REPORTS_DIR / args.segment
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "baseline_results.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
