"""Failure modes of the prediction service, exercised rather than assumed.

Every test here breaks something on purpose -- deletes an artifact, truncates
one, bumps a schema version, sends a hostile payload, fires concurrent requests
-- and asserts on what the caller actually receives. The rule being enforced
throughout: **the service either answers correctly or refuses clearly. It never
starts up and silently returns numbers it cannot stand behind, and it never
leaks an internal exception, a traceback or a filesystem path to a client.**
"""

from __future__ import annotations

import json
import logging
import re
import threading

import pytest
from fastapi.testclient import TestClient

from msa.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    ArtifactCorruptError,
    ArtifactError,
    ArtifactMissingError,
    ArtifactSchemaMismatchError,
)
from msa.serve import app as app_module
from msa.serve.app import ModelBundle, create_app

SEGMENT = "permit_to_completion"

BODY = {
    "jurisdiction": "Testville",
    "start_date": "2024-05-01",
    "state": "IL",
    "system_size_kw": 8.4,
    "reported_cost_usd": 24000,
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _meta_path(root):
    return root / SEGMENT / "metadata.json"


def _patch_meta(root, **changes):
    path = _meta_path(root)
    meta = json.loads(path.read_text())
    for k, v in changes.items():
        if v is _DELETE:
            meta.pop(k, None)
        else:
            meta[k] = v
    path.write_text(json.dumps(meta))


class _Delete:
    pass


_DELETE = _Delete()


#: Anything matching these in a response body would be an information leak.
LEAK_PATTERNS = (
    re.compile(r"/(?:workspaces|home|tmp|usr|var)/"),  # absolute filesystem paths
    re.compile(r"Traceback"),
    re.compile(r'File "'),
    re.compile(r"\.py\b"),
    re.compile(r"XGBoostError|JSONDecodeError|KeyError|ValueError\("),
)


def assert_no_leak(text: str) -> None:
    for pattern in LEAK_PATTERNS:
        assert not pattern.search(text), f"response leaks {pattern.pattern!r}: {text[:400]}"


# ---------------------------------------------------------------------------
# startup: a bad artifact must never be served
# ---------------------------------------------------------------------------


def test_missing_model_directory_raises_a_named_error(tmp_path):
    with pytest.raises(ArtifactMissingError):
        ModelBundle(SEGMENT, root=tmp_path)


def test_metadata_present_but_model_file_missing(broken_model_root):
    (broken_model_root / SEGMENT / "model.json").unlink()
    with pytest.raises(ArtifactMissingError):
        ModelBundle(SEGMENT, root=broken_model_root)


def test_missing_feature_store_is_refused(broken_model_root):
    (broken_model_root / SEGMENT / "feature_store.json").unlink()
    with pytest.raises(ArtifactMissingError):
        ModelBundle(SEGMENT, root=broken_model_root)


def test_truncated_metadata_is_refused(broken_model_root):
    path = _meta_path(broken_model_root)
    path.write_text(path.read_text()[: len(path.read_text()) // 2])
    with pytest.raises(ArtifactCorruptError):
        ModelBundle(SEGMENT, root=broken_model_root)


def test_empty_metadata_is_refused(broken_model_root):
    _meta_path(broken_model_root).write_text("")
    with pytest.raises(ArtifactCorruptError):
        ModelBundle(SEGMENT, root=broken_model_root)


def test_truncated_booster_is_refused(broken_model_root):
    path = broken_model_root / SEGMENT / "model.json"
    path.write_bytes(path.read_bytes()[:500])
    with pytest.raises(ArtifactCorruptError):
        ModelBundle(SEGMENT, root=broken_model_root)


def test_truncated_quantile_booster_is_refused(broken_model_root):
    path = broken_model_root / SEGMENT / "model_quantiles.json"
    path.write_bytes(path.read_bytes()[:500])
    with pytest.raises(ArtifactCorruptError):
        ModelBundle(SEGMENT, root=broken_model_root)


def test_truncated_feature_store_is_refused(broken_model_root):
    path = broken_model_root / SEGMENT / "feature_store.json"
    path.write_text(path.read_text()[:200])
    with pytest.raises(ArtifactCorruptError):
        ModelBundle(SEGMENT, root=broken_model_root)


def test_future_schema_version_is_refused(broken_model_root):
    _patch_meta(broken_model_root, artifact_schema_version=ARTIFACT_SCHEMA_VERSION + 1)
    with pytest.raises(ArtifactSchemaMismatchError):
        ModelBundle(SEGMENT, root=broken_model_root)


def test_unversioned_legacy_artifact_is_refused(broken_model_root):
    """A model written before the contract existed must not be served silently."""
    _patch_meta(broken_model_root, artifact_schema_version=_DELETE)
    with pytest.raises(ArtifactSchemaMismatchError) as excinfo:
        ModelBundle(SEGMENT, root=broken_model_root)
    assert "artifact_schema_version" in str(excinfo.value)


def test_feature_drift_is_refused(broken_model_root):
    """The failure this check exists for: a loadable model, wrong question.

    An artifact whose feature list differs from the code's still loads into
    XGBoost and still returns floats. Rejecting it is the only way the caller
    finds out the floats mean something else.
    """
    meta = json.loads(_meta_path(broken_model_root).read_text())
    features = list(meta["features"])
    features[0], features[1] = features[1], features[0]
    _patch_meta(broken_model_root, features=features)
    with pytest.raises(ArtifactSchemaMismatchError) as excinfo:
        ModelBundle(SEGMENT, root=broken_model_root)
    assert "feature list" in str(excinfo.value)


def test_dropped_feature_is_refused(broken_model_root):
    meta = json.loads(_meta_path(broken_model_root).read_text())
    _patch_meta(broken_model_root, features=meta["features"][:-1])
    with pytest.raises(ArtifactSchemaMismatchError):
        ModelBundle(SEGMENT, root=broken_model_root)


def test_missing_categorical_levels_are_refused(broken_model_root):
    meta = json.loads(_meta_path(broken_model_root).read_text())
    cats = dict(meta["categories"])
    cats.pop("jurisdiction")
    _patch_meta(broken_model_root, categories=cats)
    with pytest.raises(ArtifactSchemaMismatchError):
        ModelBundle(SEGMENT, root=broken_model_root)


def test_missing_required_key_is_refused(broken_model_root):
    _patch_meta(broken_model_root, horizon_days=_DELETE)
    with pytest.raises(ArtifactCorruptError):
        ModelBundle(SEGMENT, root=broken_model_root)


def test_create_app_propagates_the_artifact_error_by_default(tmp_path):
    """`require_model=True` is what every programmatic caller gets."""
    with pytest.raises(ArtifactError):
        create_app(SEGMENT, root=tmp_path)


def test_model_version_is_content_bound(model_root, broken_model_root):
    """Same bytes -> same version; different bytes -> different version."""
    from msa.artifacts import model_fingerprint

    a = model_fingerprint(model_root / SEGMENT / "model.json")
    b = model_fingerprint(broken_model_root / SEGMENT / "model.json")
    assert a == b
    path = broken_model_root / SEGMENT / "model.json"
    path.write_bytes(path.read_bytes() + b" ")
    assert model_fingerprint(path) != a


# ---------------------------------------------------------------------------
# degraded start: 503 everywhere, never a wrong answer
# ---------------------------------------------------------------------------


@pytest.fixture
def degraded_client(tmp_path):
    return TestClient(create_app(SEGMENT, root=tmp_path, require_model=False))


def test_degraded_health_is_live_but_says_so(degraded_client):
    r = degraded_client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False
    assert body["detail"]
    assert_no_leak(r.text)


def test_degraded_readiness_fails(degraded_client):
    r = degraded_client.get("/ready")
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "not_ready"
    assert body["checks"] == {"model_loaded": False, "canary_prediction": False}
    assert_no_leak(r.text)


@pytest.mark.parametrize(
    "method,path,payload",
    [
        ("post", "/predict", BODY),
        ("post", "/predict/batch", {"items": [BODY]}),
        ("get", "/metadata", None),
    ],
)
def test_degraded_prediction_endpoints_return_503(degraded_client, method, path, payload):
    r = (
        degraded_client.post(path, json=payload)
        if method == "post"
        else degraded_client.get(path)
    )
    assert r.status_code == 503
    body = r.json()
    assert body["error"] == "model_unavailable"
    assert body["detail"]
    assert body["request_id"]
    assert_no_leak(r.text)


def test_degraded_start_never_returns_a_prediction(degraded_client):
    """The property that matters: no numbers come out of a degraded service."""
    r = degraded_client.post("/predict", json=BODY)
    assert "predicted_days" not in r.json()


# ---------------------------------------------------------------------------
# readiness on a healthy service
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client(model_root):
    return TestClient(create_app(SEGMENT, root=model_root))


def test_ready_reports_a_live_canary_prediction(client):
    r = client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert body["checks"] == {"model_loaded": True, "canary_prediction": True}
    assert body["model_version"]
    assert body["trained_at"]


def test_health_publishes_version_and_training_date(client):
    body = client.get("/health").json()
    assert body["model_loaded"] is True
    assert body["artifact_schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert body["model_version"].startswith(SEGMENT)
    assert body["trained_at"]
    assert body["train_window"] and len(body["train_window"]) == 2


def test_every_prediction_carries_the_model_version(client):
    single = client.post("/predict", json=BODY).json()
    health = client.get("/health").json()
    assert single["model_version"] == health["model_version"]


# ---------------------------------------------------------------------------
# input validation: ugly payloads must be 4xx, never 500
# ---------------------------------------------------------------------------


UGLY_PAYLOADS = [
    pytest.param({}, id="empty-object"),
    pytest.param({"start_date": "2024-05-01"}, id="missing-jurisdiction"),
    pytest.param({"jurisdiction": "Testville"}, id="missing-start-date"),
    pytest.param(dict(BODY, start_date="not-a-date"), id="unparseable-date"),
    pytest.param(dict(BODY, start_date="2024-13-45"), id="impossible-date"),
    pytest.param(dict(BODY, start_date="2024-02-30"), id="non-existent-day"),
    pytest.param(dict(BODY, start_date="1799-01-01"), id="date-before-corpus"),
    pytest.param(dict(BODY, start_date="9999-01-01"), id="date-far-future"),
    pytest.param(dict(BODY, start_date=12345), id="date-wrong-type"),
    pytest.param(dict(BODY, system_size_kw=-5), id="negative-size"),
    pytest.param(dict(BODY, system_size_kw=1e9), id="absurd-size"),
    pytest.param(dict(BODY, system_size_kw="eight"), id="size-wrong-type"),
    pytest.param(dict(BODY, reported_cost_usd=-1), id="negative-cost"),
    pytest.param(dict(BODY, latitude=91), id="latitude-out-of-range"),
    pytest.param(dict(BODY, longitude=-181), id="longitude-out-of-range"),
    pytest.param(dict(BODY, battery=7), id="battery-not-boolean"),
    pytest.param(dict(BODY, state="Illinois"), id="state-not-two-letter"),
    pytest.param(dict(BODY, state="I2"), id="state-not-alphabetic"),
    pytest.param(dict(BODY, jurisdiction=""), id="empty-jurisdiction"),
    pytest.param(dict(BODY, jurisdiction="   "), id="whitespace-jurisdiction"),
    pytest.param(dict(BODY, jurisdiction=None), id="null-jurisdiction"),
    pytest.param(dict(BODY, jurisdiction=["Chicago"]), id="jurisdiction-wrong-type"),
    pytest.param(dict(BODY, jurisdiction="x" * 500), id="oversized-jurisdiction"),
    pytest.param(dict(BODY, contractor="y" * 5000), id="oversized-contractor"),
    pytest.param(dict(BODY, unknown_field=1), id="unknown-field"),
    pytest.param([BODY], id="array-instead-of-object"),
    pytest.param("just a string", id="string-instead-of-object"),
]


@pytest.mark.parametrize("payload", UGLY_PAYLOADS)
def test_malformed_single_requests_are_422_not_500(client, payload):
    r = client.post("/predict", json=payload)
    assert r.status_code == 422, r.text
    body = r.json()
    assert body["error"] == "validation_error"
    assert body["request_id"]
    assert isinstance(body["errors"], list) and body["errors"]
    assert_no_leak(r.text)


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param('{"jurisdiction": "Testville", "start_date": "2024-05-01",'
                     ' "system_size_kw": NaN}', id="nan-token"),
        pytest.param('{"jurisdiction": "Testville", "start_date": "2024-05-01",'
                     ' "reported_cost_usd": Infinity}', id="infinity-token"),
        pytest.param('{"jurisdiction": "Testville", "start_date": "2024-05-01",'
                     ' "reported_cost_usd": -Infinity}', id="negative-infinity"),
        pytest.param('{"jurisdiction": "Testville",', id="truncated-json"),
        pytest.param("", id="empty-body"),
        pytest.param("not json at all", id="not-json"),
    ],
)
def test_non_finite_and_malformed_json_are_rejected(client, raw):
    """JSON's `NaN`/`Infinity` tokens parse in Python but must not reach numpy.

    `ge=0` passes `+inf` happily, and an infinite cost propagates through
    `log1p` and `cost_per_kw` straight into a non-finite prediction.
    """
    r = client.post(
        "/predict", content=raw, headers={"content-type": "application/json"}
    )
    assert r.status_code == 422, r.text
    assert_no_leak(r.text)


BATCH_UGLY = [
    pytest.param({"items": []}, id="empty-batch"),
    pytest.param({"items": [BODY] * 1001}, id="oversized-batch"),
    pytest.param({"items": BODY}, id="items-not-a-list"),
    pytest.param({"items": [BODY, {"jurisdiction": "X"}]}, id="one-bad-item"),
    pytest.param({"projects": [BODY]}, id="wrong-key"),
    pytest.param({"items": [BODY], "extra": 1}, id="unknown-batch-field"),
    pytest.param({}, id="no-items-key"),
]


@pytest.mark.parametrize("payload", BATCH_UGLY)
def test_malformed_batch_requests_are_422_not_500(client, payload):
    r = client.post("/predict/batch", json=payload)
    assert r.status_code == 422, r.text
    assert r.json()["error"] == "validation_error"
    assert_no_leak(r.text)


def test_batch_at_the_limit_is_accepted(client):
    r = client.post("/predict/batch", json={"items": [BODY] * 1000})
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 1000


def test_oversized_body_is_rejected_before_parsing(client, monkeypatch):
    monkeypatch.setattr(app_module, "MAX_BODY_BYTES", 256)
    r = client.post("/predict/batch", json={"items": [BODY] * 50})
    assert r.status_code == 413
    body = r.json()
    assert body["error"] == "payload_too_large"
    assert body["request_id"]
    assert_no_leak(r.text)


def test_unknown_route_is_a_structured_404(client):
    r = client.get("/nope")
    assert r.status_code == 404
    assert r.json()["error"] == "not_found"
    assert_no_leak(r.text)


def test_wrong_method_is_a_structured_405(client):
    r = client.get("/predict")
    assert r.status_code == 405
    assert r.json()["error"] == "method_not_allowed"
    assert_no_leak(r.text)


def test_unknown_jurisdiction_and_contractor_still_answer(client):
    r = client.post(
        "/predict",
        json=dict(BODY, jurisdiction="Nowhere-on-Sea", contractor="NOBODY LLC"),
    )
    assert r.status_code == 200
    assert 0 <= r.json()["predicted_days"] <= r.json()["horizon_days"]


# ---------------------------------------------------------------------------
# error handling: no internals on the wire
# ---------------------------------------------------------------------------


def test_internal_failure_returns_a_bare_500_with_no_exception_text(model_root, caplog):
    """A previous review caught `detail=str(exc)` leaking. Prove it cannot recur."""
    app = create_app(SEGMENT, root=model_root)
    bundle = app.state.bundle

    class _Exploding:
        def predict(self, X):
            raise RuntimeError(
                "secret detail: /workspaces/private/model.json failed to score"
            )

    original = bundle.model
    bundle.model = _Exploding()
    try:
        with caplog.at_level(logging.ERROR):
            with TestClient(app, raise_server_exceptions=False) as tc:
                r = tc.post("/predict", json=BODY)
    finally:
        bundle.model = original

    assert r.status_code == 500
    body = r.json()
    assert body == {
        "error": "internal_error",
        "detail": "internal error",
        "request_id": body["request_id"],
    }
    assert body["request_id"]
    assert "secret detail" not in r.text
    assert_no_leak(r.text)
    # the detail is not lost -- it went to the log, keyed by the same id
    assert any("secret detail" in rec.getMessage() or
               (rec.exc_info and "secret detail" in str(rec.exc_info[1]))
               for rec in caplog.records)


def test_error_bodies_never_contain_a_filesystem_path(degraded_client):
    for path, payload in (("/predict", BODY), ("/metadata", None)):
        r = (
            degraded_client.post(path, json=payload)
            if payload
            else degraded_client.get(path)
        )
        assert_no_leak(r.text)


# ---------------------------------------------------------------------------
# correlation + structured logging
# ---------------------------------------------------------------------------


def test_request_id_is_echoed_when_supplied(client):
    r = client.post("/predict", json=BODY, headers={"X-Request-ID": "abc-123"})
    assert r.headers["X-Request-ID"] == "abc-123"


def test_request_id_is_generated_when_absent_and_is_unique(client):
    a = client.post("/predict", json=BODY).headers["X-Request-ID"]
    b = client.post("/predict", json=BODY).headers["X-Request-ID"]
    assert a and b and a != b


def test_client_supplied_request_id_is_length_capped(client):
    r = client.post("/predict", json=BODY, headers={"X-Request-ID": "z" * 5000})
    assert len(r.headers["X-Request-ID"]) <= 64


def test_logs_are_single_line_json_with_the_request_id():
    from msa.serve.logging_config import JsonFormatter, request_id_var

    token = request_id_var.set("rid-42")
    try:
        record = logging.LogRecord(
            "msa.serve.app", logging.INFO, __file__, 1, "hello %s", ("world",), None
        )
        record.path = "/predict"
        line = JsonFormatter().format(record)
    finally:
        request_id_var.reset(token)

    assert "\n" not in line
    payload = json.loads(line)
    assert payload["message"] == "hello world"
    assert payload["request_id"] == "rid-42"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "msa.serve.app"
    assert payload["path"] == "/predict"


def test_json_formatter_serialises_unserialisable_extras():
    from msa.serve.logging_config import JsonFormatter

    record = logging.LogRecord("t", logging.INFO, __file__, 1, "m", (), None)
    record.thing = object()
    payload = json.loads(JsonFormatter().format(record))
    assert isinstance(payload["thing"], str)


# ---------------------------------------------------------------------------
# concurrency
# ---------------------------------------------------------------------------


def test_model_is_loaded_once_and_shared(model_root):
    app = create_app(SEGMENT, root=model_root)
    with TestClient(app) as tc:
        seen = []

        def hit():
            tc.post("/predict", json=BODY)
            seen.append(id(app.state.bundle.model))

        threads = [threading.Thread(target=hit) for _ in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert len(set(seen)) == 1


def test_concurrent_predictions_agree_with_the_serial_answer(client):
    serial = client.post("/predict", json=BODY).json()["predicted_days"]
    results: list[float] = []
    errors: list[BaseException] = []

    def hit():
        try:
            r = client.post("/predict", json=BODY)
            assert r.status_code == 200
            results.append(r.json()["predicted_days"])
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=hit) for _ in range(32)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    assert len(results) == 32
    assert set(results) == {serial}


def test_concurrent_explains_stay_correct_and_bounded_in_memory(client):
    """TreeSHAP under load: identical answers, and RSS does not run away."""
    import resource

    payload = {"items": [BODY] * app_module.MAX_EXPLAIN_BATCH}
    # warm up so the explainer and its buffers are already allocated
    baseline = client.post("/predict/batch?explain=true", json=payload)
    assert baseline.status_code == 200
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    statuses: list[int] = []
    lock = threading.Lock()

    def hit():
        r = client.post("/predict/batch?explain=true", json=payload)
        with lock:
            statuses.append(r.status_code)

    threads = [threading.Thread(target=hit) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert statuses == [200] * 16
    # ru_maxrss is KiB on Linux. 16 concurrent explained batches must not add
    # anything like 16x a batch's working set; the semaphore caps it at
    # MAX_CONCURRENT_EXPLAIN in flight.
    assert rss_after - rss_before < 300_000, (
        f"peak RSS grew {rss_after - rss_before} KiB under 16 concurrent explains"
    )


def _measure_explain_concurrency(bundle, n_threads: int) -> int:
    """Peak simultaneous TreeSHAP calls when ``n_threads`` explain at once.

    The instrumented ``shap_values`` sleeps, so threads genuinely overlap
    rather than finishing before the next one starts -- otherwise the peak
    would be 1 regardless of whether any cap exists.
    """
    import time as _time

    in_flight = 0
    peak = 0
    lock = threading.Lock()
    real = bundle.explainer.shap_values

    def instrumented(X):
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        try:
            _time.sleep(0.05)
            return real(X)
        finally:
            with lock:
                in_flight -= 1

    bundle._explainer.shap_values = instrumented
    try:
        frame = bundle.canary
        threads = [
            threading.Thread(target=lambda: bundle.shap_values(frame))
            for _ in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        bundle._explainer.shap_values = real
    return peak


def test_explain_concurrency_is_capped(model_root):
    """The semaphore genuinely limits how many TreeSHAP calls are in flight.

    Measured twice: once through the real bounded semaphore, once with the
    bound removed. If the second run did not exceed the cap the first run's
    result would prove nothing, so both are asserted.
    """
    bundle = ModelBundle(SEGMENT, root=model_root)
    capped = _measure_explain_concurrency(bundle, 12)
    assert capped <= app_module.MAX_CONCURRENT_EXPLAIN

    bundle._explain_slots = threading.BoundedSemaphore(64)
    uncapped = _measure_explain_concurrency(bundle, 12)
    assert uncapped > app_module.MAX_CONCURRENT_EXPLAIN, (
        "the control run did not overlap, so the capped run proves nothing"
    )


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------


def test_two_bundles_from_the_same_artifact_predict_identically(model_root):
    a = ModelBundle(SEGMENT, root=model_root)
    b = ModelBundle(SEGMENT, root=model_root)
    assert a.version == b.version
    assert a.smoke_test() == b.smoke_test()


def test_repeated_requests_are_bit_identical(client):
    values = {
        client.post("/predict", json=BODY).json()["predicted_days"] for _ in range(10)
    }
    assert len(values) == 1
