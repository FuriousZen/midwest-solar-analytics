"""Tests for the parts of the pipeline that are easy to get subtly wrong:
target construction, the censoring guard, and -- most importantly -- the
guarantee that no feature can see the future.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from msa.config import MIN_TIMELINE_DAYS, SEGMENTS, TARGET, max_timeline_days
from msa.features.build import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    build_features,
    clean,
)

# --------------------------------------------------------------------------
# target construction
# --------------------------------------------------------------------------


@pytest.mark.parametrize("segment", list(SEGMENTS))
def test_target_equals_date_difference(synthetic_canonical, segment):
    start_col, end_col = SEGMENTS[segment]
    out = clean(synthetic_canonical, segment=segment, apply_censor_guard=False)
    expected = (out[end_col] - out[start_col]).dt.days
    assert (out[TARGET] == expected).all()
    assert out[TARGET].between(MIN_TIMELINE_DAYS, max_timeline_days(segment)).all()


def test_clean_drops_rows_outside_horizon(synthetic_canonical):
    df = synthetic_canonical.copy()
    df.loc[0, "completion_date"] = df.loc[0, "issue_date"] + pd.Timedelta(days=5000)
    df.loc[1, "completion_date"] = df.loc[1, "issue_date"] - pd.Timedelta(days=10)
    out = clean(df, segment="permit_to_completion", apply_censor_guard=False)
    assert "test:0" not in set(out["record_id"].str.split("|").str[0])
    assert "test:1" not in set(out["record_id"].str.split("|").str[0])


def test_censor_guard_trims_the_recent_tail(synthetic_canonical):
    seg = "permit_to_completion"
    loose = clean(synthetic_canonical, segment=seg, apply_censor_guard=False)
    tight = clean(synthetic_canonical, segment=seg, apply_censor_guard=True)
    assert len(tight) < len(loose)
    horizon = max_timeline_days(seg)
    cutoff = loose["end_date"].max() - pd.Timedelta(days=horizon)
    assert (tight["start_date"] <= cutoff).all()


def test_pooled_segment_stacks_and_tags(synthetic_canonical):
    pooled = clean(synthetic_canonical, segment="pooled", apply_censor_guard=False)
    assert set(pooled["timeline_segment"].unique()) == set(SEGMENTS)
    # a record contributing to several segments keeps distinct ids
    assert pooled["record_id"].is_unique


# --------------------------------------------------------------------------
# feature contract
# --------------------------------------------------------------------------


def test_feature_frame_shape_and_dtypes(synthetic_canonical):
    df = clean(synthetic_canonical, segment="permit_to_completion")
    X, y = build_features(df)
    assert list(X.columns) == list(FEATURES)
    assert len(X) == len(y) == len(df)
    for c in CATEGORICAL_FEATURES:
        assert isinstance(X[c].dtype, pd.CategoricalDtype)
    for c in NUMERIC_FEATURES:
        assert pd.api.types.is_numeric_dtype(X[c])


def test_no_feature_name_collisions():
    assert len(set(FEATURES)) == len(FEATURES)
    assert not set(NUMERIC_FEATURES) & set(CATEGORICAL_FEATURES)


def test_absolute_time_index_is_excluded():
    """Year / days-since-start memorise period levels and do not extrapolate."""
    assert "start_year" not in FEATURES
    assert "days_since_series_start" not in FEATURES


# --------------------------------------------------------------------------
# leakage: the property the whole project rests on
# --------------------------------------------------------------------------


def test_contractor_history_excludes_the_current_row(synthetic_canonical):
    df = clean(synthetic_canonical, segment="permit_to_completion")
    X, y = build_features(df)
    frame = pd.concat(
        [
            df[["contractor", "start_date"]].reset_index(drop=True),
            X[["contractor_prior_permits", "contractor_prior_mean_days"]],
            y.rename("y").reset_index(drop=True),
        ],
        axis=1,
    )
    for _contractor, sub in frame.dropna(subset=["contractor"]).groupby("contractor"):
        sub = sub.sort_values("start_date")
        # first job for a contractor has no history at all
        assert sub["contractor_prior_permits"].iloc[0] == 0
        assert pd.isna(sub["contractor_prior_mean_days"].iloc[0])
        # every later row's mean equals the mean of strictly earlier targets
        for k in range(1, min(len(sub), 12)):
            expected = sub["y"].iloc[:k].mean()
            assert sub["contractor_prior_mean_days"].iloc[k] == pytest.approx(expected)


def test_jurisdiction_recent_stats_use_only_already_finished_jobs(synthetic_canonical):
    df = clean(synthetic_canonical, segment="permit_to_completion")
    X, _ = build_features(df)
    col = "jurisdiction_recent_mean_days_60d"
    joined = pd.concat([df.reset_index(drop=True), X[[col]]], axis=1)
    for _juris, sub in joined.groupby("jurisdiction"):
        for i in range(0, len(sub), 17):  # sample rows to keep the test quick
            row = sub.iloc[i]
            if pd.isna(row[col]):
                continue
            window = sub[
                (sub["end_date"] < row["start_date"])
                & (sub["end_date"] >= row["start_date"] - pd.Timedelta(days=60))
            ]
            assert len(window) > 0
            assert row[col] == pytest.approx(window[TARGET].mean(), rel=1e-6)


def test_open_permit_count_is_observable_at_start_date(synthetic_canonical):
    df = clean(synthetic_canonical, segment="permit_to_completion")
    X, _ = build_features(df)
    joined = pd.concat(
        [df.reset_index(drop=True), X[["jurisdiction_open_permits"]]], axis=1
    )
    for _juris, sub in joined.groupby("jurisdiction"):
        for i in range(0, len(sub), 23):
            row = sub.iloc[i]
            expected = (
                (sub["start_date"] < row["start_date"]) & (sub["end_date"] > row["start_date"])
            ).sum()
            assert row["jurisdiction_open_permits"] == expected


def test_jurisdiction_application_counts_are_strictly_prior(synthetic_canonical):
    """Same-day siblings must not be counted: they are not observable at filing."""
    df = clean(synthetic_canonical, segment="permit_to_completion")
    X, _ = build_features(df)
    joined = pd.concat(
        [df.reset_index(drop=True), X[["jurisdiction_apps_prev_30d"]]], axis=1
    )
    for _, sub in joined.groupby("jurisdiction"):
        for i in range(0, len(sub), 13):
            row = sub.iloc[i]
            expected = (
                (sub["start_date"] < row["start_date"])
                & (sub["start_date"] >= row["start_date"] - pd.Timedelta(days=30))
            ).sum()
            assert row["jurisdiction_apps_prev_30d"] == expected


def test_shuffling_the_target_destroys_signal(synthetic_canonical):
    """A crude but decisive leakage check.

    If any feature encoded the row's own target, a model trained on a shuffled
    target would still score well. It must not.
    """
    from sklearn.model_selection import cross_val_score
    from xgboost import XGBRegressor

    df = clean(synthetic_canonical, segment="permit_to_completion")
    X, y = build_features(df)
    rng = np.random.default_rng(1)
    y_shuffled = pd.Series(rng.permutation(y.to_numpy()), index=y.index)
    model = XGBRegressor(
        enable_categorical=True, tree_method="hist", n_estimators=60, max_depth=3
    )
    scores = cross_val_score(model, X, y_shuffled, cv=3, scoring="r2")
    assert scores.mean() < 0.05, f"leakage suspected: R2={scores.mean():.3f}"


# --------------------------------------------------------------------------
# conformal calibration of the prediction interval
# --------------------------------------------------------------------------


def test_conformal_delta_lifts_coverage_towards_nominal(synthetic_canonical):
    """CQR must move empirical coverage towards the nominal 80%, not away."""
    from msa.train.data import Dataset
    from msa.train.final import (
        QUANTILES,
        calibrate_conformal,
        fit_quantiles,
        predict_quantiles,
    )

    df = clean(synthetic_canonical, segment="permit_to_completion")
    X, y = build_features(df)
    cut = int(len(X) * 0.75)
    ds = Dataset(
        X_train=X.iloc[:cut].reset_index(drop=True),
        y_train=y.iloc[:cut].reset_index(drop=True),
        X_test=X.iloc[cut:].reset_index(drop=True),
        y_test=y.iloc[cut:].reset_index(drop=True),
        meta_train=df.iloc[:cut].reset_index(drop=True),
        meta_test=df.iloc[cut:].reset_index(drop=True),
        split_date=df["start_date"].iloc[cut],
        categories={c: list(map(str, X[c].cat.categories)) for c in CATEGORICAL_FEATURES},
        segment="permit_to_completion",
    )
    params = {"n_estimators": 60, "max_depth": 3}
    horizon = float(max_timeline_days("permit_to_completion"))
    delta = calibrate_conformal(ds, params, log_target=False, horizon=horizon)
    assert np.isfinite(delta)

    q = predict_quantiles(
        fit_quantiles(ds, params, False), ds.X_test, False, horizon
    )
    truth = ds.y_test.to_numpy()
    raw = ((truth >= q[:, 0]) & (truth <= q[:, 2])).mean()
    cal = (
        (truth >= np.clip(q[:, 0] - delta, 0, None)) & (truth <= q[:, 2] + delta)
    ).mean()
    nominal = QUANTILES[2] - QUANTILES[0]
    assert abs(cal - nominal) <= abs(raw - nominal) + 1e-9


def test_back_transform_never_returns_infinity():
    """A log-target booster can emit a score whose expm1 overflows to +inf.

    That killed trial 185 of the first 200-trial study, so the back-transform is
    clamped in log space and the behaviour is pinned here.
    """
    from msa.train.tune import back_transform

    raw = np.array([-5.0, 0.0, 3.0, 1e3, 1e30, np.inf, np.nan])
    out = back_transform(raw, log_target=True, horizon=365.0)
    assert np.isfinite(out).all()
    assert (out >= 0).all() and (out <= 365.0).all()

    out2 = back_transform(np.array([-10.0, 50.0, 1e12]), log_target=False, horizon=120.0)
    assert np.allclose(out2, [0.0, 50.0, 120.0])
