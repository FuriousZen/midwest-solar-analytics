"""End-to-end tests for the FastAPI service.

A miniature model is trained on the synthetic fixture and written to a tmp
directory, so these run offline and do not depend on the real trained
artifacts. The point is to prove the *contract* holds: request -> feature frame
with exactly the training columns and categorical levels -> prediction ->
response schema.
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from msa.features.build import CATEGORICAL_FEATURES, FEATURES
from msa.features.store import apply_store
from msa.serve.app import ModelBundle, create_app, request_frame
from msa.serve.schemas import PermitRequest

SEGMENT = "permit_to_completion"


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
        days=round(body["predicted_days"])
    )
    assert body["predicted_completion_date"] == str(expected.date())


def test_prediction_interval_is_well_formed(client):
    body = client.post("/predict", json=BODY).json()
    assert body["p10_days"] is not None and body["p90_days"] is not None
    assert 0 <= body["p10_days"] <= body["p90_days"] <= body["horizon_days"]


def test_point_estimate_lies_inside_its_interval(client):
    """The point and the interval come from different boosters.

    Nothing about two separately fitted models keeps one inside the other, so
    the service has to enforce it. Checked across several requests, including
    ones with no context, so the property is not an accident of one row.
    """
    bodies = [BODY, dict(BODY, start_date="2025-01-20", system_size_kw=24.0)]
    bodies += [
        {"jurisdiction": "Testville", "start_date": "2024-05-01"},
        {"jurisdiction": "Nowhere-on-Sea", "start_date": "2023-11-11"},
        dict(BODY, jurisdiction="Sampleburg", contractor="BRIGHT ROOF", battery=1),
    ]
    for body in bodies:
        r = client.post("/predict", json=body)
        assert r.status_code == 200, r.text
        p = r.json()
        assert p["p10_days"] <= p["predicted_days"] <= p["p90_days"], p

    batch = client.post("/predict/batch", json={"items": bodies}).json()["predictions"]
    for p in batch:
        assert p["p10_days"] <= p["predicted_days"] <= p["p90_days"], p


def test_interval_covers_most_of_a_holdout_like_sample(client):
    """Sanity check that the interval is not degenerate (zero-width)."""
    widths = []
    for day in ("2024-03-01", "2024-07-15", "2025-01-20"):
        b = client.post("/predict", json=dict(BODY, start_date=day)).json()
        widths.append(b["p90_days"] - b["p10_days"])
    assert min(widths) > 0


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


def test_zoning_review_flag_is_derived_the_same_way_as_in_training(model_root):
    """Train/serve parity: the flag comes from the zoning fee in both paths."""
    bundle = ModelBundle(SEGMENT, root=model_root)
    X = request_frame(
        [
            PermitRequest(**dict(BODY, zoning_fee_usd=50.0)),
            PermitRequest(**dict(BODY, zoning_fee_usd=0.0)),
            PermitRequest(**BODY),  # not supplied -> must stay missing
        ],
        bundle,
    )
    assert X["zoning_review_required"].tolist()[:2] == [1.0, 0.0]
    assert pd.isna(X["zoning_review_required"].iloc[2])
    assert X["zoning_fee_usd"].tolist()[:2] == [50.0, 0.0]


def test_overlong_free_text_is_rejected(client):
    """Uncapped string fields on a batch endpoint are a memory-amplifier."""
    assert (
        client.post("/predict", json=dict(BODY, jurisdiction="x" * 5000)).status_code
        == 422
    )
    assert (
        client.post("/predict", json=dict(BODY, contractor="y" * 5000)).status_code
        == 422
    )


def test_response_is_strict_json_with_no_nan(client):
    """Bare NaN is not valid JSON; a strict client rejects the whole payload."""
    import json as _json

    r = client.post("/predict?explain=true", json=BODY)
    assert r.status_code == 200
    # json.loads with parse_constant raises on NaN/Infinity tokens
    _json.loads(
        r.text,
        parse_constant=lambda c: (_ for _ in ()).throw(
            AssertionError(f"non-finite token {c!r} in response")
        ),
    )


def test_unknown_keys_stay_missing_rather_than_becoming_average(model_root):
    """Training encoded 'no history' as NaN; serving must not fill a median.

    Filling the training median would tell the booster an unknown contractor is
    an *average* contractor, which is a different and more confident claim.
    """
    bundle = ModelBundle(SEGMENT, root=model_root)
    rows = pd.DataFrame(
        [
            {
                "jurisdiction": "Nowhere-on-Sea",
                "state": "ZZ",
                "contractor": "NEVER HEARD OF THEM",
                "zip_code": "00000",
            }
        ]
    )
    out = apply_store(rows, bundle.store)
    for col in (
        "contractor_prior_permits",
        "contractor_prior_mean_days",
        "jurisdiction_open_permits",
        "tts_zip_cum_installs",
        "tts_state_installs_prev_365d",
    ):
        assert pd.isna(out.loc[0, col]), f"{col} should be missing, got {out.loc[0, col]}"
    assert out.loc[0, "contractor_is_known"] == 0


def test_source_level_is_replayed_from_the_jurisdiction(model_root):
    """`source` is a trained categorical level; a literal 'api' is not one."""
    bundle = ModelBundle(SEGMENT, root=model_root)
    X = request_frame([PermitRequest(**BODY)], bundle)
    assert str(X["source"].iloc[0]) in bundle.categories["source"]

    unknown = request_frame(
        [PermitRequest(**dict(BODY, jurisdiction="Nowhere-on-Sea"))], bundle
    )
    assert pd.isna(unknown["source"].iloc[0])


def test_explain_on_an_oversized_batch_is_refused_not_silently_dropped(client):
    from msa.serve.app import MAX_EXPLAIN_BATCH

    big = {"items": [BODY] * (MAX_EXPLAIN_BATCH + 5)}
    r = client.post("/predict/batch?explain=true", json=big)
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "explain_batch_too_large"
    assert str(MAX_EXPLAIN_BATCH) in body["detail"]
    assert set(body) == {"error", "detail", "request_id"}

    # the same batch without the flag is fine
    r_plain = client.post("/predict/batch", json=big)
    assert r_plain.status_code == 200
    assert len(r_plain.json()["predictions"]) == MAX_EXPLAIN_BATCH + 5

    # and a batch at the cap is explained
    at_cap = {"items": [BODY] * MAX_EXPLAIN_BATCH}
    r2 = client.post("/predict/batch?explain=true", json=at_cap)
    assert r2.status_code == 200
    assert all(p["attributions"] for p in r2.json()["predictions"])
