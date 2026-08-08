"""End-to-end tests for the FastAPI service.

A miniature model is trained on the synthetic fixture and written to a tmp
directory, so these run offline and do not depend on the real trained
artifacts. The point is to prove the *contract* holds: request -> feature frame
with exactly the training columns and categorical levels -> prediction ->
response schema.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from msa.features.build import CATEGORICAL_FEATURES, FEATURES, build_features, clean
from msa.features.store import apply_store, build_store, save_store
from msa.serve.app import ModelBundle, create_app, request_frame
from msa.serve.schemas import PermitRequest

SEGMENT = "permit_to_completion"


@pytest.fixture(scope="module")
def model_root(tmp_path_factory, request):
    """Train a small real model on synthetic records and persist it."""
    from xgboost import XGBRegressor

    from msa.config import CANONICAL_COLUMNS, max_timeline_days

    rng = np.random.default_rng(0)
    n = 400
    start = pd.Timestamp("2021-01-01") + pd.to_timedelta(
        rng.integers(0, 900, n), unit="D"
    )
    review = rng.integers(1, 40, n)
    build_days = rng.integers(5, 200, n)
    df = pd.DataFrame(
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

    cleaned = clean(df, segment=SEGMENT, apply_censor_guard=False)
    X, y = build_features(cleaned)
    model = XGBRegressor(
        enable_categorical=True, tree_method="hist", n_estimators=40, max_depth=3
    )
    model.fit(X, y)

    root = tmp_path_factory.mktemp("models")
    d = root / SEGMENT
    d.mkdir()
    model.save_model(str(d / "model.json"))
    meta = cleaned[["record_id", "source", "jurisdiction", "state", "start_date", "end_date"]]
    save_store(build_store(X, meta, cleaned), d / "feature_store.json")
    (d / "metadata.json").write_text(
        json.dumps(
            {
                "segment": SEGMENT,
                "horizon_days": max_timeline_days(SEGMENT),
                "features": list(FEATURES),
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
                "metrics": {"cv_mae": 1.0},
            }
        )
    )
    return root


@pytest.fixture(scope="module")
def client(model_root):
    return TestClient(create_app(SEGMENT, root=model_root))


BODY = {
    "jurisdiction": "Testville",
    "start_date": "2024-05-01",
    "state": "IL",
    "system_size_kw": 8.4,
    "reported_cost_usd": 24000,
    "sector": "residential",
    "review_track": "Express",
    "contractor": "ACME SOLAR",
    "zip_code": "60601",
    "mount_type": "roof",
    "roof_type": "pitched",
    "battery": 0,
}


def test_health_reports_the_loaded_model(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["segment"] == SEGMENT
    assert body["n_features"] == len(FEATURES)
    assert body["trained_on"] > 0
    assert "Testville" in body["jurisdictions"]


def test_predict_returns_a_sane_duration(client):
    r = client.post("/predict", json=BODY)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["predicted_days"] <= body["horizon_days"]
    assert body["segment"] == SEGMENT
    assert body["attributions"] is None
    # completion date must be start_date + predicted_days
    expected = pd.Timestamp("2024-05-01") + pd.Timedelta(
        days=int(round(body["predicted_days"]))
    )
    assert body["predicted_completion_date"] == str(expected.date())


def test_predict_with_explain_returns_ranked_attributions(client):
    r = client.post("/predict?explain=true", json=BODY)
    assert r.status_code == 200
    attrs = r.json()["attributions"]
    assert attrs and len(attrs) <= 8
    magnitudes = [abs(a["shap"]) for a in attrs]
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert all(a["feature"] in FEATURES for a in attrs)


def test_minimal_body_is_accepted(client):
    r = client.post(
        "/predict", json={"jurisdiction": "Testville", "start_date": "2024-05-01"}
    )
    assert r.status_code == 200
    assert r.json()["predicted_days"] >= 0


def test_batch_matches_single_predictions(client):
    r = client.post("/predict/batch", json={"items": [BODY, BODY, dict(BODY, jurisdiction="Sampleburg")]})
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert len(preds) == 3
    single = client.post("/predict", json=BODY).json()["predicted_days"]
    assert preds[0]["predicted_days"] == pytest.approx(single)


def test_invalid_payloads_are_rejected(client):
    assert client.post("/predict", json={"start_date": "2024-05-01"}).status_code == 422
    assert client.post("/predict", json={"jurisdiction": "X"}).status_code == 422
    bad_size = dict(BODY, system_size_kw=-5)
    assert client.post("/predict", json=bad_size).status_code == 422
    assert client.post("/predict/batch", json={"items": []}).status_code == 422


def test_unknown_jurisdiction_does_not_crash(client):
    r = client.post("/predict", json=dict(BODY, jurisdiction="Nowhere-on-Sea"))
    assert r.status_code == 200
    assert r.json()["predicted_days"] >= 0


def test_metrics_endpoint_tracks_latency(client):
    client.post("/predict", json=BODY)
    body = client.get("/metrics").json()
    assert body["predict"] > 0
    assert body["latency_ms"]["p95"] >= body["latency_ms"]["p50"] >= 0


# --------------------------------------------------------------------------
# the contract that makes the service correct
# --------------------------------------------------------------------------


def test_request_frame_matches_training_columns_and_levels(model_root):
    bundle = ModelBundle(SEGMENT, root=model_root)
    X = request_frame([PermitRequest(**BODY)], bundle)
    assert list(X.columns) == list(FEATURES)
    for c in CATEGORICAL_FEATURES:
        assert list(map(str, X[c].cat.categories)) == bundle.categories[c]


def test_feature_store_recovers_contractor_history(model_root):
    bundle = ModelBundle(SEGMENT, root=model_root)
    rows = pd.DataFrame(
        [
            {
                "jurisdiction": "Testville",
                "state": "IL",
                "contractor": "ACME SOLAR",
                "zip_code": "60601",
            },
            {
                "jurisdiction": "Testville",
                "state": "IL",
                "contractor": "NEVER HEARD OF THEM",
                "zip_code": "60601",
            },
        ]
    )
    out = apply_store(rows, bundle.store)
    assert out["contractor_is_known"].tolist() == [1, 0]
    assert out.loc[0, "contractor_prior_permits"] > 0
    assert out.loc[0, "jurisdiction_open_permits"] is not None
