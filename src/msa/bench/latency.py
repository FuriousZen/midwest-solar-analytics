"""Measure the real end-to-end latency distribution of the prediction service.

Two modes:

``--mode inprocess`` (default)
    Drives the ASGI app through ``httpx.ASGITransport``. This measures
    validation + feature assembly + booster inference without a network hop --
    the fair number for "how long does the service take to answer".

``--mode http --url http://127.0.0.1:8000``
    Fires real HTTP requests at a running uvicorn worker, so the number
    includes the socket, the HTTP parser and the event loop.

Both modes report p50/p90/p95/p99, run a warm-up that is discarded, and write
``reports/latency_<mode>.json``. Payloads are sampled from the held-out set so
the request distribution matches reality rather than one hand-picked example.

Run::

    python -m msa.bench.latency --requests 2000
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import statistics
import time
from datetime import date

import numpy as np

from msa.config import DEFAULT_SEGMENT, REPORTS_DIR, SEGMENTS

LOG = logging.getLogger(__name__)


def sample_payloads(segment: str, n: int) -> list[dict]:
    """Build request bodies from real held-out rows."""
    import pandas as pd

    from msa.features.build import clean
    from msa.train.data import load_raw

    df = clean(load_raw(), segment=segment)
    df = df.sort_values("start_date").tail(max(n, 200))
    rng = np.random.default_rng(7)
    idx = rng.integers(0, len(df), size=n)
    rows = df.iloc[idx]

    payloads = []
    for _, r in rows.iterrows():
        body: dict = {
            "jurisdiction": str(r["jurisdiction"]),
            "start_date": str(pd.Timestamp(r["start_date"]).date()),
            "state": str(r["state"]),
        }
        for src, dst in (
            ("system_size_kw", "system_size_kw"),
            ("reported_cost_usd", "reported_cost_usd"),
            ("permit_fee_usd", "permit_fee_usd"),
            ("battery", "battery"),
            ("latitude", "latitude"),
            ("longitude", "longitude"),
        ):
            v = r.get(src)
            if v is not None and not pd.isna(v):
                body[dst] = float(v)
        for src in ("sector", "review_track", "permit_class", "district",
                    "contractor", "zip_code", "mount_type", "roof_type"):
            v = r.get(src)
            if isinstance(v, str) and v:
                body[src] = v
        payloads.append(body)
    return payloads


def percentiles(samples: list[float]) -> dict[str, float]:
    a = np.asarray(samples, dtype=float)
    return {
        "n": int(a.size),
        "mean_ms": float(a.mean()),
        "p50_ms": float(np.percentile(a, 50)),
        "p90_ms": float(np.percentile(a, 90)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
        "max_ms": float(a.max()),
        "min_ms": float(a.min()),
        "stdev_ms": float(statistics.pstdev(a.tolist())),
    }


def bench_inprocess(segment: str, payloads: list[dict], warmup: int) -> dict:
    import httpx

    from msa.serve.app import create_app

    app = create_app(segment)
    transport = httpx.ASGITransport(app=app)
    with httpx.Client(transport=transport, base_url="http://bench") as client:
        for p in payloads[:warmup]:
            client.post("/predict", json=p)
        samples: list[float] = []
        for p in payloads:
            t0 = time.perf_counter()
            resp = client.post("/predict", json=p)
            samples.append((time.perf_counter() - t0) * 1000.0)
            if resp.status_code != 200:
                raise RuntimeError(f"{resp.status_code}: {resp.text[:300]}")
    return percentiles(samples)


def bench_http(url: str, payloads: list[dict], warmup: int) -> dict:
    import httpx

    with httpx.Client(base_url=url, timeout=30.0) as client:
        health = client.get("/health")
        health.raise_for_status()
        for p in payloads[:warmup]:
            client.post("/predict", json=p)
        samples: list[float] = []
        for p in payloads:
            t0 = time.perf_counter()
            resp = client.post("/predict", json=p)
            samples.append((time.perf_counter() - t0) * 1000.0)
            resp.raise_for_status()
    return percentiles(samples)


def bench_batch(segment: str, payloads: list[dict], size: int = 100) -> dict:
    import httpx

    from msa.serve.app import create_app

    app = create_app(segment)
    transport = httpx.ASGITransport(app=app)
    samples: list[float] = []
    with httpx.Client(transport=transport, base_url="http://bench") as client:
        client.post("/predict/batch", json={"items": payloads[:size]})
        for i in range(0, len(payloads) - size, size):
            batch = payloads[i : i + size]
            t0 = time.perf_counter()
            resp = client.post("/predict/batch", json={"items": batch})
            samples.append((time.perf_counter() - t0) * 1000.0)
            resp.raise_for_status()
    stats = percentiles(samples)
    stats["batch_size"] = size
    stats["per_item_p50_ms"] = stats["p50_ms"] / size
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment", default=DEFAULT_SEGMENT, choices=list(SEGMENTS))
    ap.add_argument("--requests", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--mode", choices=["inprocess", "http"], default="inprocess")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--explain", action="store_true", help="also bench ?explain=true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    payloads = sample_payloads(args.segment, args.requests)
    LOG.info("benchmarking %d requests (%s)", len(payloads), args.mode)

    if args.mode == "inprocess":
        single = bench_inprocess(args.segment, payloads, args.warmup)
        batch = bench_batch(args.segment, payloads)
    else:
        single = bench_http(args.url, payloads, args.warmup)
        batch = None

    result = {
        "segment": args.segment,
        "mode": args.mode,
        "measured_on": str(date.today()),
        "host": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or "n/a",
        },
        "single_request": single,
        "batch_request": batch,
    }
    out = REPORTS_DIR / f"latency_{args.mode}.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    LOG.info(
        "p50=%.2f ms  p95=%.2f ms  p99=%.2f ms  -> %s",
        single["p50_ms"],
        single["p95_ms"],
        single["p99_ms"],
        out,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
