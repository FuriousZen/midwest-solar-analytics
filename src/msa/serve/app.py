"""FastAPI prediction service for solar installation timelines.

Endpoints
---------
``GET  /health``           model identity, training window, feature-store date
``GET  /metadata``         full model card: metrics, features, jurisdictions
``POST /predict``          one project  (``?explain=true`` adds SHAP attributions)
``POST /predict/batch``    up to 1000 projects in one call
``GET  /metrics``          in-process request counters and latency percentiles

Run::

    uvicorn msa.serve.app:app --host 127.0.0.1 --port 8000

The model, its metadata and the context feature store are loaded once at import
time and held in memory; a request touches no disk and no network.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from msa.config import DEFAULT_SEGMENT, MODELS_DIR
from msa.features.build import (
    CATEGORICAL_FEATURES,
    FEATURES,
    as_model_frame,
    back_transform,
)
from msa.features.store import apply_store, load_store
from msa.serve.schemas import (
    Attribution,
    BatchRequest,
    BatchResponse,
    HealthResponse,
    PermitRequest,
    PredictionResponse,
)

LOG = logging.getLogger(__name__)

SEGMENT = os.environ.get("MSA_SERVE_SEGMENT", DEFAULT_SEGMENT)
TOP_ATTRIBUTIONS = 8


class ModelBundle:
    """Everything needed to answer a request, loaded once."""

    def __init__(self, segment: str, root: Path = MODELS_DIR):
        d = root / segment
        meta_path = d / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{meta_path} not found -- run `python -m msa.train.final "
                f"--segment {segment}` first"
            )
        self.segment = segment
        self.metadata: dict[str, Any] = json.loads(meta_path.read_text())
        self.model = xgb.XGBRegressor()
        self.model.load_model(str(d / "model.json"))
        self.booster = self.model.get_booster()
        self.store = load_store(d / "feature_store.json")
        self.quantiles = list(self.metadata.get("quantiles") or [])
        self.conformal_delta = float(self.metadata.get("conformal_delta_days") or 0.0)
        qpath = d / "model_quantiles.json"
        self.qmodel = None
        if qpath.exists():
            self.qmodel = xgb.XGBRegressor()
            self.qmodel.load_model(str(qpath))
        self.log_target = bool(self.metadata.get("log1p_target"))
        self.categories = self.metadata["categories"]
        self.horizon = int(self.metadata["horizon_days"])
        self.version = self.metadata.get("model_version") or _version_of(d)
        self._explainer = None

    @property
    def explainer(self):
        # TreeExplainer construction costs ~100 ms, so build it lazily: the hot
        # path (?explain not set) never pays for it.
        if self._explainer is None:
            import shap

            self._explainer = shap.TreeExplainer(self.model)
        return self._explainer


def _version_of(d: Path) -> str:
    stat = (d / "model.json").stat()
    return f"{d.name}-{int(stat.st_mtime)}"


def request_frame(items: list[PermitRequest], bundle: ModelBundle) -> pd.DataFrame:
    """Turn request bodies into the exact feature matrix the booster expects."""
    raw = pd.DataFrame([i.model_dump() for i in items])
    raw["start_date"] = pd.to_datetime(raw["start_date"])
    raw["source"] = "api"
    raw["jurisdiction"] = raw["jurisdiction"].astype("string")

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


def create_app(segment: str = SEGMENT, root: Path = MODELS_DIR) -> FastAPI:
    bundle = ModelBundle(segment, root)
    latencies: deque[float] = deque(maxlen=5000)
    counters = {"predict": 0, "predict_batch": 0, "errors": 0}

    app = FastAPI(
        title="Midwest Solar Analytics -- timeline prediction",
        version="0.1.0",
        description=(
            "Predicts how many days a distributed solar project will take, from "
            "public Midwest permit registers. Segment: " + segment
        ),
    )
    app.state.bundle = bundle
    app.state.latencies = latencies
    app.state.counters = counters

    def _predict(items: list[PermitRequest], explain: bool) -> list[PredictionResponse]:
        X = request_frame(items, bundle)
        preds = back_transform(
            bundle.model.predict(X), bundle.log_target, bundle.horizon
        )

        # TreeSHAP is computed once for the whole frame; calling it per row
        # would make an explained batch quadratic in the batch size.
        shap_values = (
            np.asarray(bundle.explainer.shap_values(X)) if explain else None
        )

        bounds = None
        if bundle.qmodel is not None:
            q = np.atleast_2d(bundle.qmodel.predict(X))
            if q.shape[0] != len(X):
                q = q.T
            q = np.sort(back_transform(q, bundle.log_target, bundle.horizon), axis=1)
            # split-conformal widening estimated during training
            q[:, 0] -= bundle.conformal_delta
            q[:, -1] += bundle.conformal_delta
            bounds = np.clip(q, 0.0, bundle.horizon)

        out = []
        for i, item in enumerate(items):
            days = float(preds[i])
            out.append(
                PredictionResponse(
                    predicted_days=round(days, 2),
                    p10_days=round(float(bounds[i, 0]), 2) if bounds is not None else None,
                    p90_days=round(float(bounds[i, -1]), 2) if bounds is not None else None,
                    predicted_completion_date=item.start_date
                    + timedelta(days=int(round(days))),
                    horizon_days=bundle.horizon,
                    segment=bundle.segment,
                    model_version=bundle.version,
                    attributions=(
                        _attributions(X, i, shap_values) if explain else None
                    ),
                )
            )
        return out

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        m = bundle.metadata
        return HealthResponse(
            status="ok",
            segment=bundle.segment,
            model_version=bundle.version,
            n_features=len(FEATURES),
            trained_on=int(m["n_train"]),
            train_window=[m["train_start"], m["train_end"]],
            jurisdictions=m["jurisdictions"],
        )

    @app.get("/metadata")
    def metadata() -> JSONResponse:
        payload = {
            k: v for k, v in bundle.metadata.items() if k != "categories"
        }
        payload["feature_store_as_of"] = bundle.store.get("as_of")
        payload["known_contractors"] = len(bundle.store.get("contractor", {}))
        return JSONResponse(payload)

    @app.post("/predict", response_model=PredictionResponse)
    def predict(
        item: PermitRequest,
        explain: bool = Query(False, description="attach SHAP attributions"),
    ) -> PredictionResponse:
        t0 = time.perf_counter()
        try:
            result = _predict([item], explain)[0]
        except Exception as exc:  # noqa: BLE001
            counters["errors"] += 1
            LOG.exception("prediction failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        latencies.append((time.perf_counter() - t0) * 1000.0)
        counters["predict"] += 1
        return result

    @app.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(
        body: BatchRequest,
        explain: bool = Query(False),
    ) -> BatchResponse:
        t0 = time.perf_counter()
        try:
            results = _predict(body.items, explain)
        except Exception as exc:  # noqa: BLE001
            counters["errors"] += 1
            LOG.exception("batch prediction failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        latencies.append((time.perf_counter() - t0) * 1000.0)
        counters["predict_batch"] += 1
        return BatchResponse(predictions=results)

    @app.get("/metrics")
    def metrics() -> JSONResponse:
        arr = np.array(latencies) if latencies else np.array([0.0])
        return JSONResponse(
            {
                **counters,
                "observed_requests": len(latencies),
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


class _LazyASGI:
    """Defer model loading until the server actually starts.

    ``uvicorn msa.serve.app:app`` needs an importable ASGI object, but importing
    this module must not require a trained model on disk -- otherwise the test
    suite and the docs tooling cannot import it at all. The bundle is built on
    the first request/lifespan event instead.
    """

    def __init__(self, segment: str = SEGMENT):
        self._segment = segment
        self._app: FastAPI | None = None

    @property
    def app(self) -> FastAPI:
        if self._app is None:
            self._app = create_app(self._segment)
        return self._app

    async def __call__(self, scope, receive, send):  # pragma: no cover - ASGI glue
        await self.app(scope, receive, send)


app = _LazyASGI()
