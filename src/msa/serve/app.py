"""FastAPI prediction service for solar installation timelines.

Endpoints
---------
``GET  /health``           liveness + model identity; answers even when degraded
``GET  /ready``            readiness: the model is loaded *and* just answered a
                           canary request
``GET  /metadata``         full model card: metrics, features, jurisdictions
``POST /predict``          one project  (``?explain=true`` adds SHAP attributions)
``POST /predict/batch``    up to 1000 projects in one call
``GET  /metrics``          in-process request counters and latency percentiles

Run::

    uvicorn msa.serve.app:app --host 127.0.0.1 --port 8000

Startup contract
----------------
The model directory is validated against :mod:`msa.artifacts` before the
service will answer anything. A missing, truncated, or schema-incompatible
artifact is never served: by default the process **refuses to start** and says
why. Set ``MSA_ALLOW_DEGRADED_START=1`` to start anyway, in which case
``/health`` reports ``degraded``, ``/ready`` returns 503, and every prediction
endpoint returns 503 with a short reason. There is no configuration in which
the service starts and quietly returns numbers from an artifact it could not
verify.

The model, its metadata and the context feature store are loaded once and held
in memory; a request touches no disk and no network. XGBoost's prediction
functions are documented thread-safe for ``gbtree`` boosters as long as nothing
mutates the model concurrently, which nothing here does -- so the single shared
booster serves the whole worker thread-pool without a lock.
See https://xgboost.readthedocs.io/en/stable/prediction.html
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from msa.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    ArtifactCorruptError,
    ArtifactError,
    ArtifactMissingError,
    build_model_version,
    read_metadata,
    validate_metadata,
)
from msa.config import DEFAULT_SEGMENT, MODELS_DIR
from msa.features.build import (
    CATEGORICAL_FEATURES,
    FEATURES,
    as_model_frame,
    back_transform,
)
from msa.features.store import apply_store, load_store
from msa.serve.logging_config import (
    configure_logging,
    new_request_id,
    request_id_var,
)
from msa.serve.schemas import (
    MAX_EXPLAIN_BATCH_ITEMS,
    Attribution,
    BatchRequest,
    BatchResponse,
    ErrorResponse,
    HealthResponse,
    PermitRequest,
    PredictionResponse,
    ReadinessResponse,
)

LOG = logging.getLogger(__name__)

SEGMENT = os.environ.get("MSA_SERVE_SEGMENT", DEFAULT_SEGMENT)
TOP_ATTRIBUTIONS = 8

#: TreeSHAP over a full batch is far more expensive than prediction, so
#: ``?explain=true`` is only honoured for batches up to this size. Larger
#: batches still return predictions; they just do not carry attributions.
MAX_EXPLAIN_BATCH = MAX_EXPLAIN_BATCH_ITEMS

#: TreeSHAP allocates an ``(n_rows, n_features + 1)`` float64 matrix per call
#: *and* XGBoost allocates its own internal buffers. With an unbounded worker
#: pool, N concurrent explained batches means N simultaneous allocations. Cap
#: the number in flight so peak memory is bounded by a constant rather than by
#: how many clients happen to call at once.
MAX_CONCURRENT_EXPLAIN = int(os.environ.get("MSA_MAX_CONCURRENT_EXPLAIN", "2"))

#: Largest request body accepted, in bytes. 1000 permits of ~600 bytes each is
#: comfortably under this; anything larger is rejected before it is parsed.
MAX_BODY_BYTES = int(os.environ.get("MSA_MAX_BODY_BYTES", str(4 * 1024 * 1024)))

#: Header carrying the correlation id, in and out.
REQUEST_ID_HEADER = "X-Request-ID"


def _degraded_start_allowed() -> bool:
    return os.environ.get("MSA_ALLOW_DEGRADED_START", "").lower() in {"1", "true", "yes"}


class BodySizeLimitMiddleware:
    """Bound the request body even when the client never declares its size.

    The ``Content-Length`` check one layer in rejects an oversized body before a
    byte of it is read, which handles every ordinary client. It does nothing
    about ``Transfer-Encoding: chunked``, where there is no declared length and
    the ASGI server will buffer whatever arrives.

    So this layer reads the body itself, counting as it goes, and answers 413
    the moment the cap is passed -- without ever calling the application. Peak
    memory per request is therefore bounded by the cap whether or not the client
    declares a length. The buffered chunks are replayed to the application
    afterwards so nothing downstream can tell the difference.

    Written as raw ASGI rather than ``BaseHTTPMiddleware`` for two reasons: it
    has to wrap ``receive``, which ``BaseHTTPMiddleware`` does not expose; and
    an exception raised from inside a wrapped ``receive`` does not propagate
    cleanly through ``BaseHTTPMiddleware``'s body pump -- it surfaces at the
    route layer as an unrecognised error and FastAPI renders it
    ``400 There was an error parsing the body``. Answering directly sidesteps
    the whole question.
    """

    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def _reject(self, send) -> None:
        body = json.dumps(
            {
                "error": "payload_too_large",
                "detail": f"request body exceeds the {self.max_bytes} byte limit",
                "request_id": request_id_var.get(),
            }
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        buffered: list[dict] = []
        total = 0
        more = True
        while more:
            message = await receive()
            if message.get("type") != "http.request":
                buffered.append(message)
                break
            total += len(message.get("body", b""))
            if total > self.max_bytes:
                LOG.warning(
                    "rejecting oversized request body",
                    extra={"bytes_read": total, "limit": self.max_bytes},
                )
                await self._reject(send)
                return
            buffered.append(message)
            more = bool(message.get("more_body", False))

        queued = iter(buffered)

        async def replay():
            try:
                return next(queued)
            except StopIteration:
                return await receive()

        await self.app(scope, replay, send)


class ModelBundle:
    """Everything needed to answer a request, loaded and verified once.

    The constructor is the only place that trusts the filesystem. If it
    returns, the artifact has been checked against the code's own feature
    contract and has answered a canary prediction.
    """

    def __init__(self, segment: str, root: Path = MODELS_DIR):
        d = root / segment
        meta_path = d / "metadata.json"
        model_path = d / "model.json"

        self.metadata: dict[str, Any] = read_metadata(meta_path)
        validate_metadata(
            self.metadata,
            expected_features=FEATURES,
            expected_categoricals=CATEGORICAL_FEATURES,
            path=meta_path,
        )

        if not model_path.exists():
            raise ArtifactMissingError(
                f"{meta_path} exists but {model_path} does not -- the model "
                f"directory is incomplete. Refit with `python -m msa.train.final "
                f"--segment {segment}`."
            )

        self.segment = segment
        self.model = xgb.XGBRegressor()
        try:
            self.model.load_model(str(model_path))
        except Exception as exc:
            raise ArtifactCorruptError(
                f"{model_path} could not be loaded as an XGBoost booster "
                f"(truncated or not a booster file): {exc}"
            ) from exc
        self.booster = self.model.get_booster()

        n_booster_features = self.booster.num_features()
        if n_booster_features != len(FEATURES):
            raise ArtifactCorruptError(
                f"{model_path} expects {n_booster_features} features but the "
                f"feature contract has {len(FEATURES)}"
            )

        store_path = d / "feature_store.json"
        if not store_path.exists():
            raise ArtifactMissingError(
                f"{store_path} not found -- the model directory is incomplete"
            )
        try:
            self.store = load_store(store_path)
        except json.JSONDecodeError as exc:
            raise ArtifactCorruptError(
                f"{store_path} is not valid JSON (truncated write?): {exc}"
            ) from exc

        self.quantiles = list(self.metadata.get("quantiles") or [])
        self.conformal_delta = float(self.metadata.get("conformal_delta_days") or 0.0)
        qpath = d / "model_quantiles.json"
        self.qmodel = None
        if qpath.exists():
            self.qmodel = xgb.XGBRegressor()
            try:
                self.qmodel.load_model(str(qpath))
            except Exception as exc:
                raise ArtifactCorruptError(
                    f"{qpath} could not be loaded as an XGBoost booster: {exc}"
                ) from exc

        self.log_target = bool(self.metadata.get("log1p_target"))
        self.categories = self.metadata["categories"]
        self.horizon = int(self.metadata["horizon_days"])
        self.artifact_schema_version = int(self.metadata["artifact_schema_version"])
        self.trained_at: str = str(self.metadata.get("trained_at") or "")
        self.version: str = str(
            self.metadata.get("model_version")
            or build_model_version(segment, model_path, self.trained_at or "0000-00-00")
        )
        self._explainer = None
        self._explainer_lock = threading.Lock()
        self._explain_slots = threading.BoundedSemaphore(MAX_CONCURRENT_EXPLAIN)

        self.canary = self._build_canary()
        self.smoke_test()

    # -- canary ------------------------------------------------------------

    def _build_canary(self) -> pd.DataFrame:
        """A one-row feature frame used to prove the booster can actually score.

        Built from the first jurisdiction the model was trained on and the last
        day of its training window, so it exercises the real feature-store join
        rather than an all-missing row.
        """
        jurisdictions = self.metadata.get("jurisdictions") or ["unknown"]
        item = PermitRequest(
            jurisdiction=str(jurisdictions[0])[:120],
            start_date=pd.Timestamp(self.metadata["train_end"]).date(),
        )
        return request_frame([item], self)

    def smoke_test(self) -> float:
        """Score the canary row. Raises if the loaded model cannot predict."""
        try:
            value = float(
                back_transform(
                    self.model.predict(self.canary), self.log_target, self.horizon
                )[0]
            )
        except Exception as exc:
            raise ArtifactCorruptError(
                f"loaded model for segment {self.segment!r} could not score a "
                f"canary row: {exc}"
            ) from exc
        if not np.isfinite(value):
            raise ArtifactCorruptError(
                f"loaded model for segment {self.segment!r} returned a "
                f"non-finite prediction for the canary row"
            )
        return value

    # -- explainer ---------------------------------------------------------

    @property
    def explainer(self):
        # TreeExplainer construction costs ~100 ms, so build it lazily: the hot
        # path (?explain not set) never pays for it. The lock stops two
        # concurrent first-explain requests each building their own.
        if self._explainer is None:
            with self._explainer_lock:
                if self._explainer is None:
                    import shap

                    self._explainer = shap.TreeExplainer(self.model)
        return self._explainer

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """TreeSHAP for ``X``, with the number of concurrent calls bounded."""
        with self._explain_slots:
            return np.asarray(self.explainer.shap_values(X))

    def identity(self) -> dict[str, Any]:
        return {
            "segment": self.segment,
            "model_version": self.version,
            "artifact_schema_version": self.artifact_schema_version,
            "trained_at": self.trained_at or None,
        }


def request_frame(items: list[PermitRequest], bundle: ModelBundle) -> pd.DataFrame:
    """Turn request bodies into the exact feature matrix the booster expects."""
    raw = pd.DataFrame([i.model_dump() for i in items])
    raw["start_date"] = pd.to_datetime(raw["start_date"])
    raw["jurisdiction"] = raw["jurisdiction"].astype("string")

    # `source` is filled in by the feature store from the jurisdiction, so that
    # the categorical level matches one the booster was trained on. An unknown
    # jurisdiction leaves it missing, which is the honest encoding.
    frame = apply_store(raw, bundle.store)

    d = frame["start_date"]
    frame["start_month"] = d.dt.month
    frame["start_dayofweek"] = d.dt.dayofweek
    frame["start_dayofyear"] = d.dt.dayofyear
    frame["start_quarter"] = d.dt.quarter
    frame["start_is_month_end"] = d.dt.is_month_end.astype(int)
    frame["start_week_of_year"] = d.dt.isocalendar().week.astype(int)

    size = pd.to_numeric(frame.get("system_size_kw"), errors="coerce")
    cost = pd.to_numeric(frame.get("reported_cost_usd"), errors="coerce")
    frame["log_system_size_kw"] = np.log1p(size)
    frame["log_reported_cost_usd"] = np.log1p(cost)
    frame["cost_per_kw"] = (cost / size).replace([np.inf, -np.inf], np.nan)
    # `zoning_review_required` is derived from the zoning fee at training time
    # (msa.features.build._application_features); deriving it the same way here
    # rather than hard-coding NaN keeps the training and serving definitions
    # identical. Missing stays missing -- XGBoost splits on NaN natively, and a
    # register that publishes no fee breakdown had NaN during training too.
    zoning = pd.to_numeric(frame.get("zoning_fee_usd"), errors="coerce")
    frame["zoning_review_required"] = np.where(zoning.isna(), np.nan, (zoning > 0).astype(float))
    frame["timeline_segment"] = bundle.segment

    return as_model_frame(frame, bundle.categories)


def _attributions(
    X: pd.DataFrame, row: int, shap_values: np.ndarray
) -> list[Attribution]:
    """Top-|SHAP| drivers for one row of an already-computed attribution matrix."""
    v = shap_values[row]
    order = np.argsort(-np.abs(v))[:TOP_ATTRIBUTIONS]
    out = []
    for j in order:
        name = X.columns[j]
        raw = X.iloc[row, j]
        if name in CATEGORICAL_FEATURES:
            val: float | str | None = None if pd.isna(raw) else str(raw)
        else:
            val = None if pd.isna(raw) else float(raw)
        out.append(Attribution(feature=name, value=val, shap=float(v[j])))
    return out


def _request_id(request: Request | None = None) -> str | None:
    """The correlation id for the current request.

    Read from ``request.state`` first and the context variable second. The
    500 handler runs *outside* the correlation middleware (Starlette's
    ``ServerErrorMiddleware`` is the outermost layer), by which point the
    context variable has already been reset -- but ``scope["state"]`` is the
    same dict the middleware wrote to, so the id survives.
    """
    if request is not None:
        rid = getattr(request.state, "request_id", None)
        if rid:
            return str(rid)
    return request_id_var.get()


def _error(
    status_code: int,
    code: str,
    detail: str,
    request: Request | None = None,
    request_id: str | None = None,
) -> JSONResponse:
    """The single place an error body is constructed.

    Nothing derived from an exception object ever reaches this function's
    ``detail`` -- callers pass a fixed string. Tracebacks, file paths and
    library messages go to the log, keyed by the request id.
    """
    body = ErrorResponse(
        error=code, detail=detail, request_id=request_id or _request_id(request)
    )
    return JSONResponse(status_code=status_code, content=body.model_dump())


class _Counters:
    """Request counters that stay consistent under a multi-threaded worker pool."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values = {
            "predict": 0,
            "predict_batch": 0,
            "errors": 0,
            "rejected_unavailable": 0,
        }

    def incr(self, key: str, by: int = 1) -> None:
        with self._lock:
            self._values[key] = self._values.get(key, 0) + by

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._values)


def create_app(
    segment: str = SEGMENT,
    root: Path = MODELS_DIR,
    *,
    require_model: bool = True,
) -> FastAPI:
    """Build the ASGI application.

    ``require_model=True`` (the default, and what every programmatic caller
    wants) raises :class:`msa.artifacts.ArtifactError` if the model directory
    is missing, corrupt or schema-incompatible. ``require_model=False`` builds
    a service that starts, reports itself degraded, and returns 503 from every
    prediction endpoint -- used by the ASGI entrypoint when
    ``MSA_ALLOW_DEGRADED_START=1``.
    """
    configure_logging()

    bundle: ModelBundle | None = None
    load_error: ArtifactError | None = None
    try:
        bundle = ModelBundle(segment, root)
    except ArtifactError as exc:
        if require_model:
            raise
        load_error = exc
        LOG.critical(
            "model artifact for segment %s could not be loaded; starting in "
            "DEGRADED mode -- every prediction endpoint will return 503. %s",
            segment,
            exc,
            extra={"segment": segment, "artifact_error": type(exc).__name__},
        )

    latencies: deque[float] = deque(maxlen=5000)
    counters = _Counters()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if bundle is not None:
            LOG.info(
                "model loaded",
                extra={
                    **bundle.identity(),
                    "n_features": len(FEATURES),
                    "n_train": bundle.metadata.get("n_train"),
                    "feature_store_as_of": bundle.store.get("as_of"),
                },
            )
        else:
            LOG.critical(
                "serving DEGRADED: %s",
                load_error.public_reason if load_error else "unknown",
                extra={"segment": segment},
            )
        yield

    app = FastAPI(
        title="Midwest Solar Analytics -- timeline prediction",
        version="0.1.0",
        lifespan=lifespan,
        description=(
            "Predicts how many days a distributed solar project will take, from "
            "public Midwest permit registers. Segment: " + segment
        ),
    )
    app.state.bundle = bundle
    app.state.load_error = load_error
    app.state.latencies = latencies
    app.state.counters = counters

    # outermost of our own layers: caps a chunked body that declares no length
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=MAX_BODY_BYTES)

    # -- cross-cutting middleware -----------------------------------------

    @app.middleware("http")
    async def correlate_and_log(request: Request, call_next):
        rid = request.headers.get(REQUEST_ID_HEADER) or new_request_id()
        # a client-supplied id is echoed but never trusted into a log field
        # unbounded in length
        rid = rid[:64]
        token = request_id_var.set(rid)
        # also on scope["state"], which outlives this middleware and is
        # therefore still readable from the 500 handler above it
        request.state.request_id = rid
        t0 = time.perf_counter()
        try:
            declared = request.headers.get("content-length")
            if declared is not None and declared.isdigit() and int(declared) > MAX_BODY_BYTES:
                counters.incr("errors")
                response = _error(
                    413,
                    "payload_too_large",
                    f"request body exceeds the {MAX_BODY_BYTES} byte limit",
                    request_id=rid,
                )
            else:
                response = await call_next(request)
            response.headers[REQUEST_ID_HEADER] = rid
            return response
        finally:
            LOG.info(
                "request",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": round((time.perf_counter() - t0) * 1000.0, 3),
                    "request_id": rid,
                },
            )
            request_id_var.reset(token)

    # -- error handling ----------------------------------------------------

    @app.exception_handler(RequestValidationError)
    async def _on_validation_error(request: Request, exc: RequestValidationError):
        counters.incr("errors")
        # `loc`/`msg`/`type` describe the caller's own payload and are safe to
        # return. `input` and `ctx` are dropped: `ctx` can carry library
        # internals and `input` can echo an oversized body straight back.
        errors = [
            {
                "loc": [str(p) for p in e.get("loc", ())],
                "msg": str(e.get("msg", "invalid")),
                "type": str(e.get("type", "value_error")),
            }
            for e in exc.errors()[:50]
        ]
        LOG.info("request rejected by validation", extra={"n_errors": len(errors)})
        payload = ErrorResponse(
            error="validation_error",
            detail="request body failed validation",
            request_id=request_id_var.get(),
        ).model_dump()
        payload["errors"] = errors
        return JSONResponse(status_code=422, content=payload)

    @app.exception_handler(StarletteHTTPException)
    async def _on_http_error(request: Request, exc: StarletteHTTPException):
        detail = exc.detail if isinstance(exc.detail, str) else "request failed"
        code = {
            404: "not_found",
            405: "method_not_allowed",
            413: "payload_too_large",
            503: "model_unavailable",
        }.get(exc.status_code, "http_error")
        return _error(exc.status_code, code, detail, request)

    @app.exception_handler(Exception)
    async def _on_unhandled(request: Request, exc: Exception):
        counters.incr("errors")
        LOG.exception(
            "unhandled error", extra={"path": request.url.path}
        )
        return _error(500, "internal_error", "internal error", request)

    # -- prediction --------------------------------------------------------

    def _require_bundle() -> ModelBundle:
        if bundle is None:
            counters.incr("rejected_unavailable")
            reason = load_error.public_reason if load_error else "model not loaded"
            raise StarletteHTTPException(status_code=503, detail=reason)
        return bundle

    def _predict(items: list[PermitRequest], explain: bool) -> list[PredictionResponse]:
        b = _require_bundle()
        X = request_frame(items, b)
        preds = back_transform(b.model.predict(X), b.log_target, b.horizon)

        # TreeSHAP is computed once for the whole frame; calling it per row
        # would make an explained batch quadratic in the batch size.
        explain = explain and len(X) <= MAX_EXPLAIN_BATCH
        shap_values = b.shap_values(X) if explain else None

        bounds = None
        if b.qmodel is not None:
            q = np.atleast_2d(b.qmodel.predict(X))
            if q.shape[0] != len(X):
                q = q.T
            q = np.sort(back_transform(q, b.log_target, b.horizon), axis=1)
            # split-conformal widening estimated during training
            q[:, 0] -= b.conformal_delta
            q[:, -1] += b.conformal_delta
            bounds = np.clip(q, 0.0, b.horizon)

        out = []
        for i, item in enumerate(items):
            days = float(preds[i])
            out.append(
                PredictionResponse(
                    predicted_days=round(days, 2),
                    p10_days=round(float(bounds[i, 0]), 2) if bounds is not None else None,
                    p90_days=round(float(bounds[i, -1]), 2) if bounds is not None else None,
                    predicted_completion_date=item.start_date
                    + timedelta(days=round(days)),
                    horizon_days=b.horizon,
                    segment=b.segment,
                    model_version=b.version,
                    attributions=(
                        _attributions(X, i, shap_values) if explain else None
                    ),
                )
            )
        return out

    # -- endpoints ---------------------------------------------------------

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Liveness. 200 whenever the process is up, degraded or not."""
        if bundle is None:
            return HealthResponse(
                status="degraded",
                segment=segment,
                model_loaded=False,
                detail=load_error.public_reason if load_error else "model not loaded",
            )
        m = bundle.metadata
        return HealthResponse(
            status="ok",
            segment=bundle.segment,
            model_loaded=True,
            model_version=bundle.version,
            artifact_schema_version=bundle.artifact_schema_version,
            trained_at=bundle.trained_at or None,
            n_features=len(FEATURES),
            trained_on=int(m["n_train"]),
            train_window=[m["train_start"], m["train_end"]],
            jurisdictions=m["jurisdictions"],
        )

    @app.get(
        "/ready",
        response_model=ReadinessResponse,
        responses={503: {"model": ReadinessResponse}},
    )
    def ready():
        """Readiness. Verifies the model is loaded *and* still scores a row."""
        if bundle is None:
            return JSONResponse(
                status_code=503,
                content=ReadinessResponse(
                    status="not_ready",
                    segment=segment,
                    model_loaded=False,
                    checks={"model_loaded": False, "canary_prediction": False},
                    detail=load_error.public_reason if load_error else "model not loaded",
                ).model_dump(),
            )
        try:
            bundle.smoke_test()
        except ArtifactError:
            LOG.exception("readiness canary failed")
            return JSONResponse(
                status_code=503,
                content=ReadinessResponse(
                    status="not_ready",
                    segment=bundle.segment,
                    model_loaded=True,
                    model_version=bundle.version,
                    trained_at=bundle.trained_at or None,
                    checks={"model_loaded": True, "canary_prediction": False},
                    detail="loaded model failed its canary prediction",
                ).model_dump(),
            )
        return ReadinessResponse(
            status="ready",
            segment=bundle.segment,
            model_loaded=True,
            model_version=bundle.version,
            trained_at=bundle.trained_at or None,
            checks={"model_loaded": True, "canary_prediction": True},
        )

    @app.get("/metadata")
    def metadata() -> JSONResponse:
        b = _require_bundle()
        payload = {k: v for k, v in b.metadata.items() if k != "categories"}
        payload["feature_store_as_of"] = b.store.get("as_of")
        payload["known_contractors"] = len(b.store.get("contractor", {}))
        payload["model_version"] = b.version
        return JSONResponse(payload)

    @app.post(
        "/predict",
        response_model=PredictionResponse,
        responses={
            422: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
    )
    def predict(
        item: PermitRequest,
        explain: bool = Query(False, description="attach SHAP attributions"),
    ) -> PredictionResponse:
        t0 = time.perf_counter()
        result = _predict([item], explain)[0]
        latencies.append((time.perf_counter() - t0) * 1000.0)
        counters.incr("predict")
        return result

    @app.post(
        "/predict/batch",
        response_model=BatchResponse,
        responses={
            413: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
    )
    def predict_batch(
        body: BatchRequest,
        explain: bool = Query(False),
    ) -> BatchResponse:
        t0 = time.perf_counter()
        results = _predict(body.items, explain)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        counters.incr("predict_batch")
        return BatchResponse(predictions=results)

    @app.get("/metrics")
    def metrics() -> JSONResponse:
        snapshot = list(latencies)
        arr = np.array(snapshot) if snapshot else np.array([0.0])
        return JSONResponse(
            {
                **counters.snapshot(),
                "model_loaded": bundle is not None,
                "model_version": bundle.version if bundle else None,
                "observed_requests": len(snapshot),
                "latency_ms": {
                    "p50": float(np.percentile(arr, 50)),
                    "p90": float(np.percentile(arr, 90)),
                    "p95": float(np.percentile(arr, 95)),
                    "p99": float(np.percentile(arr, 99)),
                    "max": float(arr.max()),
                },
            }
        )

    return app


class _ServiceASGI:
    """The ASGI entrypoint, so that importing this module needs no model on disk.

    ``uvicorn msa.serve.app:app`` needs an importable ASGI object, but importing
    this module must not require a trained model -- otherwise the test suite and
    the docs tooling cannot import it at all. The app is therefore built on the
    first ASGI event, which under uvicorn is the ``lifespan`` startup: a bad
    artifact aborts startup with a clear message *before* the socket accepts a
    single request.
    """

    def __init__(self, segment: str = SEGMENT, root: Path | None = None):
        self._segment = segment
        # resolved lazily: MODELS_DIR is fixed at import time, but a test (or a
        # sidecar deployment) needs to point the entrypoint somewhere else
        self._root = root
        self._app: FastAPI | None = None
        self._lock = threading.Lock()

    def build(self) -> FastAPI:
        if self._app is None:
            with self._lock:
                if self._app is None:
                    configure_logging()
                    allow_degraded = _degraded_start_allowed()
                    try:
                        self._app = create_app(
                            self._segment,
                            root=self._root or MODELS_DIR,
                            require_model=not allow_degraded,
                        )
                    except ArtifactError as exc:
                        LOG.critical(
                            "refusing to start: %s. Set MSA_ALLOW_DEGRADED_START=1 "
                            "to start anyway and serve 503 from the prediction "
                            "endpoints.",
                            exc,
                            extra={"segment": self._segment},
                        )
                        raise
        return self._app

    @property
    def app(self) -> FastAPI:
        return self.build()

    async def _lifespan(self, scope, receive, send) -> None:
        """Handle the lifespan scope so a bad artifact actually stops the server.

        Raising here does **not** work: uvicorn interprets an exception during
        the lifespan scope as "this app does not implement the lifespan
        protocol", logs ``ASGI 'lifespan' protocol appears unsupported`` and
        starts serving anyway -- which is precisely the outcome this service
        must never have. The ASGI spec's way to refuse startup is to send
        ``lifespan.startup.failed``; uvicorn then logs the message and exits.
        Verified against uvicorn 0.52.1.
        """
        message = await receive()
        if message["type"] != "lifespan.startup":  # pragma: no cover - defensive
            return
        try:
            inner = self.build()
        except Exception as exc:
            await send(
                {
                    "type": "lifespan.startup.failed",
                    "message": (
                        f"{type(exc).__name__}: {exc} "
                        f"(set MSA_ALLOW_DEGRADED_START=1 to start anyway and "
                        f"serve 503 from the prediction endpoints)"
                    ),
                }
            )
            return

        # hand the already-consumed startup message to the inner app so its own
        # lifespan runs exactly as it would have
        replayed = False

        async def _receive():
            nonlocal replayed
            if not replayed:
                replayed = True
                return message
            return await receive()

        await inner(scope, _receive, send)

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            await self._lifespan(scope, receive, send)
            return
        await self.build()(scope, receive, send)


app = _ServiceASGI()

__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "MAX_CONCURRENT_EXPLAIN",
    "MAX_EXPLAIN_BATCH",
    "BodySizeLimitMiddleware",
    "ModelBundle",
    "app",
    "create_app",
    "request_frame",
]
