"""Measure what absolute time indices do to generalisation.

`msa.features.build` deliberately omits `start_year` and
`days_since_series_start`. This script produces the evidence for that decision
by fitting the same model with and without them and comparing random-fold CV
against the chronological holdout. Writes
``reports/ablation_time_features.json``.

Run::

    python -m scripts.ablate_time_features
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from msa.config import RANDOM_SEED, REPORTS_DIR
from msa.train.baseline import make_cv
from msa.train.data import load_dataset
from msa.train.metrics import regression_metrics

LOG = logging.getLogger(__name__)


def _add_time_indices(X: pd.DataFrame, meta: pd.DataFrame, origin) -> pd.DataFrame:
    out = X.copy()
    d = pd.to_datetime(meta["start_date"])
    out["start_year"] = d.dt.year.to_numpy()
    out["days_since_series_start"] = (d - origin).dt.days.to_numpy()
    return out


def evaluate(X_tr, y_tr, X_te, y_te) -> dict:
    params = dict(
        enable_categorical=True,
        tree_method="hist",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=0,
    )
    oof = np.empty(len(y_tr))
    for tr, va in make_cv().split(X_tr):
        m = XGBRegressor(**params)
        m.fit(X_tr.iloc[tr], y_tr.iloc[tr])
        oof[va] = m.predict(X_tr.iloc[va])
    m = XGBRegressor(**params)
    m.fit(X_tr, y_tr)
    return {
        **regression_metrics(y_tr, oof, prefix="cv_"),
        **regression_metrics(y_te, m.predict(X_te), prefix="test_"),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    results = {}
    for segment in ("application_to_issue", "permit_to_completion"):
        ds = load_dataset(segment=segment)
        origin = pd.to_datetime(ds.meta_train["start_date"]).min()
        without = evaluate(ds.X_train, ds.y_train, ds.X_test, ds.y_test)
        with_ = evaluate(
            _add_time_indices(ds.X_train, ds.meta_train, origin),
            ds.y_train,
            _add_time_indices(ds.X_test, ds.meta_test, origin),
            ds.y_test,
        )
        results[segment] = {
            "model": "XGBRegressor() stock hyper-parameters",
            "n_train": ds.n_train,
            "n_test": ds.n_test,
            "with_absolute_time_indices": with_,
            "without_absolute_time_indices": without,
            "delta_test_r2": without["test_r2"] - with_["test_r2"],
            "delta_cv_r2": without["cv_r2"] - with_["cv_r2"],
        }
        LOG.info(
            "[%s] with time idx: cv_r2=%.4f test_r2=%.4f | without: cv_r2=%.4f test_r2=%.4f",
            segment,
            with_["cv_r2"],
            with_["test_r2"],
            without["cv_r2"],
            without["test_r2"],
        )
    path = REPORTS_DIR / "ablation_time_features.json"
    path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
