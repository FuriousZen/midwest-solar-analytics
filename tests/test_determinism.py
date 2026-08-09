"""Seeds must actually make runs reproducible.

Every number in the documentation is quoted as if re-running the command would
produce it again. That is only true if the seeded paths are genuinely
deterministic -- the CV folds, the booster fit, the chronological split, and
the content hash that becomes the model version.

Verified at the repository scale as well as here: two independent
``python -m msa.train.final`` runs of both segments, in separate processes,
produced byte-identical boosters --

    permit_to_completion-20260809-7c548d88b141   (run 1 and run 2)
    application_to_issue-20260809-187cc3a37e74   (run 1 and run 2)

-- and identical metrics to every reported decimal. Reproduce with::

    make final SEGMENT=permit_to_completion   # twice; compare model_version
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from msa.config import RANDOM_SEED
from msa.features.build import build_features, clean
from msa.train.baseline import default_xgb, make_cv
from msa.train.data import _holdout_mask
from msa.train.tune import cv_evaluate, make_model

SEGMENT = "permit_to_completion"

PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 60,
    "learning_rate": 0.1,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}


def _xy(synthetic_canonical):
    cleaned = clean(synthetic_canonical, segment=SEGMENT, apply_censor_guard=False)
    X, y = build_features(cleaned)
    return X, y, cleaned


def test_cv_folds_are_identical_across_constructions():
    a = list(make_cv().split(np.zeros((500, 2))))
    b = list(make_cv().split(np.zeros((500, 2))))
    for (tr_a, te_a), (tr_b, te_b) in zip(a, b, strict=True):
        np.testing.assert_array_equal(tr_a, tr_b)
        np.testing.assert_array_equal(te_a, te_b)


def test_feature_build_is_deterministic(synthetic_canonical):
    Xa, ya, _ = _xy(synthetic_canonical)
    Xb, yb, _ = _xy(synthetic_canonical)
    pd.testing.assert_frame_equal(Xa, Xb)
    pd.testing.assert_series_equal(ya, yb)


def test_two_fits_with_the_same_seed_produce_the_same_booster(synthetic_canonical):
    """Byte-identical, not merely close: this is what the version hash promises."""
    X, y, _ = _xy(synthetic_canonical)
    a = make_model(PARAMS)
    a.fit(X, y, verbose=False)
    b = make_model(PARAMS)
    b.fit(X, y, verbose=False)
    assert a.get_booster().save_raw("json") == b.get_booster().save_raw("json")
    np.testing.assert_array_equal(a.predict(X), b.predict(X))


def test_stock_baseline_is_deterministic_too(synthetic_canonical):
    X, y, _ = _xy(synthetic_canonical)
    a = default_xgb()
    a.fit(X, y)
    b = default_xgb()
    b.fit(X, y)
    np.testing.assert_array_equal(a.predict(X), b.predict(X))


def test_cv_evaluate_is_reproducible(synthetic_canonical):
    X, y, _ = _xy(synthetic_canonical)
    agg_a, _, oof_a = cv_evaluate(PARAMS, X, y, make_cv(), False, 365.0)
    agg_b, _, oof_b = cv_evaluate(PARAMS, X, y, make_cv(), False, 365.0)
    assert agg_a == agg_b
    np.testing.assert_array_equal(oof_a, oof_b)


def test_chronological_split_is_stable(synthetic_canonical):
    cleaned = clean(synthetic_canonical, segment=SEGMENT, apply_censor_guard=False)
    meta = cleaned.sort_values(["start_date", "record_id"]).reset_index(drop=True)[
        ["record_id", "jurisdiction", "start_date"]
    ]
    mask_a, date_a = _holdout_mask(meta, 0.2, "by_jurisdiction")
    mask_b, date_b = _holdout_mask(meta, 0.2, "by_jurisdiction")
    pd.testing.assert_series_equal(mask_a, mask_b)
    assert date_a == date_b
    # and the split is by date, not at random: no training row starts after the
    # earliest test row within the same jurisdiction
    frame = meta.assign(is_test=mask_a)
    for _juris, sub in frame.groupby("jurisdiction"):
        if sub["is_test"].any() and (~sub["is_test"]).any():
            assert sub.loc[~sub["is_test"], "start_date"].max() <= sub.loc[
                sub["is_test"], "start_date"
            ].min()


def test_model_fingerprint_is_a_pure_function_of_bytes(tmp_path):
    from msa.artifacts import model_fingerprint

    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_bytes(b"\x00" * 5000)
    b.write_bytes(b"\x00" * 5000)
    assert model_fingerprint(a) == model_fingerprint(b)
    b.write_bytes(b"\x00" * 4999 + b"\x01")
    assert model_fingerprint(a) != model_fingerprint(b)


def test_the_seed_is_a_single_declared_constant():
    """Nothing may reach for a random default; every seeded call uses this one."""
    assert isinstance(RANDOM_SEED, int)
    assert make_cv().random_state == RANDOM_SEED
    assert make_model({}).get_params()["random_state"] == RANDOM_SEED
    assert default_xgb().get_params()["random_state"] == RANDOM_SEED
