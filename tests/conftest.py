"""Shared fixtures.

Tests never hit the network. Anything that needs canonical records builds them
from ``synthetic_canonical``, a hand-written frame with known properties, used
only to exercise transformation logic. It is never used for training or for any
number that appears in the README.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from msa.config import CANONICAL_COLUMNS


@pytest.fixture
def synthetic_canonical() -> pd.DataFrame:
    """Deterministic canonical records for unit-testing transformations."""
    rng = np.random.default_rng(0)
    n = 400
    start = pd.Timestamp("2021-01-01") + pd.to_timedelta(
        rng.integers(0, 900, n), unit="D"
    )
    review = rng.integers(1, 40, n)
    build = rng.integers(5, 200, n)
    contractors = np.array(["ACME SOLAR", "BRIGHT ROOF", "CEDAR PV", None], dtype=object)

    df = pd.DataFrame(
        {
            "record_id": [f"test:{i}" for i in range(n)],
            "source": "test_source",
            "jurisdiction": rng.choice(["Testville", "Sampleburg"], n),
            "state": "IL",
            "application_date": start,
            "issue_date": start + pd.to_timedelta(review, unit="D"),
            "completion_date": start + pd.to_timedelta(review + build, unit="D"),
            "system_size_kw": rng.uniform(3, 25, n),
            "reported_cost_usd": rng.uniform(8000, 80000, n),
            "permit_fee_usd": rng.uniform(50, 900, n),
            "sector": rng.choice(["residential", "commercial"], n),
            "review_track": rng.choice(["Express", "Standard"], n),
            "contractor": rng.choice(contractors, n),
            "zip_code": rng.choice(["60601", "60602", "55401"], n),
            "latitude": rng.uniform(41, 45, n),
            "longitude": rng.uniform(-93, -87, n),
            "mount_type": "roof",
            "roof_type": rng.choice(["flat", "pitched"], n),
            "battery": rng.integers(0, 2, n),
            "permit_class": rng.choice(["Alteration", "New"], n),
            "district": rng.choice(["ward-1", "ward-2"], n),
            "building_fee_usd": rng.uniform(0, 500, n),
            "zoning_fee_usd": rng.choice([0.0, 50.0], n),
            "other_fee_usd": rng.uniform(0, 300, n),
            "n_parties": rng.integers(1, 6, n),
            "has_expeditor": rng.integers(0, 2, n),
            "has_engineer": rng.integers(0, 2, n),
            "has_design_professional": rng.integers(0, 2, n),
            "description_length": rng.integers(20, 400, n),
        }
    )
    return df[list(CANONICAL_COLUMNS)]


# ---------------------------------------------------------------------------
# A real, tiny, on-disk model directory
# ---------------------------------------------------------------------------
#
# The serving tests need an artifact that satisfies the full
# ``msa.artifacts`` contract: schema version, feature list, categorical
# levels, booster, quantile booster, feature store. Building it here (rather
# than in one test module) lets the hardening tests corrupt *copies* of it
# without disturbing the happy-path suite.


def _write_model_dir(dest, segment: str) -> None:
    """Fit a miniature model on synthetic records and persist a valid artifact."""
    import json

    from xgboost import XGBRegressor

    from msa.artifacts import ARTIFACT_SCHEMA_VERSION, build_model_version
    from msa.config import RANDOM_SEED, max_timeline_days
    from msa.features.build import (
        CATEGORICAL_FEATURES,
        FEATURES,
        build_features,
        clean,
    )
    from msa.features.store import build_store, save_store

    df = _synthetic_frame()
    cleaned = clean(df, segment=segment, apply_censor_guard=False)
    X, y = build_features(cleaned)

    model = XGBRegressor(
        enable_categorical=True,
        tree_method="hist",
        n_estimators=40,
        max_depth=3,
        random_state=RANDOM_SEED,
    )
    model.fit(X, y)

    qmodel = XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=np.array([0.1, 0.5, 0.9]),
        enable_categorical=True,
        tree_method="hist",
        n_estimators=40,
        max_depth=3,
        random_state=RANDOM_SEED,
    )
    qmodel.fit(X, y)

    dest.mkdir(parents=True, exist_ok=True)
    model.save_model(str(dest / "model.json"))
    qmodel.save_model(str(dest / "model_quantiles.json"))
    meta_cols = ["record_id", "source", "jurisdiction", "state", "start_date", "end_date"]
    save_store(
        build_store(X, cleaned[meta_cols], cleaned), dest / "feature_store.json"
    )
    trained_at = "2025-01-01T00:00:00+00:00"
    (dest / "metadata.json").write_text(
        json.dumps(
            {
                "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                "trained_at": trained_at,
                "model_version": build_model_version(
                    segment, dest / "model.json", trained_at
                ),
                "segment": segment,
                "horizon_days": max_timeline_days(segment),
                "features": list(FEATURES),
                "numeric_features": [
                    f for f in FEATURES if f not in CATEGORICAL_FEATURES
                ],
                "categorical_features": list(CATEGORICAL_FEATURES),
                "categories": {
                    c: list(map(str, X[c].cat.categories)) for c in CATEGORICAL_FEATURES
                },
                "log1p_target": False,
                "n_train": len(X),
                "n_test": 0,
                "train_start": str(cleaned["start_date"].min().date()),
                "train_end": str(cleaned["start_date"].max().date()),
                "test_start": str(cleaned["start_date"].max().date()),
                "test_end": str(cleaned["start_date"].max().date()),
                "jurisdictions": sorted(cleaned["jurisdiction"].unique().tolist()),
                "seed": 0,
                "metrics": {"cv_mae": 1.0},
                "quantiles": [0.1, 0.5, 0.9],
                "conformal_delta_days": 0.0,
            },
            indent=2,
        )
    )


def _synthetic_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 400
    start = pd.Timestamp("2021-01-01") + pd.to_timedelta(
        rng.integers(0, 900, n), unit="D"
    )
    review = rng.integers(1, 40, n)
    build_days = rng.integers(5, 200, n)
    return pd.DataFrame(
        {
            "record_id": [f"test:{i}" for i in range(n)],
            "source": "test_source",
            "jurisdiction": rng.choice(["Testville", "Sampleburg"], n),
            "state": "IL",
            "application_date": start,
            "issue_date": start + pd.to_timedelta(review, unit="D"),
            "completion_date": start + pd.to_timedelta(review + build_days, unit="D"),
            "system_size_kw": rng.uniform(3, 25, n),
            "reported_cost_usd": rng.uniform(8000, 80000, n),
            "permit_fee_usd": rng.uniform(50, 900, n),
            "sector": rng.choice(["residential", "commercial"], n),
            "review_track": rng.choice(["Express", "Standard"], n),
            "contractor": rng.choice(["ACME SOLAR", "BRIGHT ROOF", None], n),
            "zip_code": rng.choice(["60601", "55401"], n),
            "latitude": rng.uniform(41, 45, n),
            "longitude": rng.uniform(-93, -87, n),
            "mount_type": "roof",
            "roof_type": rng.choice(["flat", "pitched"], n),
            "battery": rng.integers(0, 2, n),
            "permit_class": rng.choice(["Alteration", "New"], n),
            "district": rng.choice(["ward-1", "ward-2"], n),
            "building_fee_usd": rng.uniform(0, 500, n),
            "zoning_fee_usd": rng.choice([0.0, 50.0], n),
            "other_fee_usd": rng.uniform(0, 300, n),
            "n_parties": rng.integers(1, 6, n),
            "has_expeditor": rng.integers(0, 2, n),
            "has_engineer": rng.integers(0, 2, n),
            "has_design_professional": rng.integers(0, 2, n),
            "description_length": rng.integers(20, 400, n),
        }
    )[list(CANONICAL_COLUMNS)]


SERVE_SEGMENT = "permit_to_completion"


@pytest.fixture(scope="session")
def model_root(tmp_path_factory):
    """A valid model directory for ``permit_to_completion``, built once."""
    root = tmp_path_factory.mktemp("models")
    _write_model_dir(root / SERVE_SEGMENT, SERVE_SEGMENT)
    return root


@pytest.fixture
def broken_model_root(tmp_path, model_root):
    """A writable copy of the good artifact, for corrupting in tests."""
    import shutil

    dest = tmp_path / "models"
    shutil.copytree(model_root, dest)
    return dest
