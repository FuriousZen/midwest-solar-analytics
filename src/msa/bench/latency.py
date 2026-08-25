"""Measure the real end-to-end latency distribution of the prediction service.

Two modes:

``--mode inprocess`` (default)
    Drives the ASGI app through ``httpx.ASGITransport``. This measures
    validation + feature assembly + booster inference without a network hop:
    the fair number for "how long does the service take to answer".

``--mode http --url http://127.0.0.1:8000``
    Fires real HTTP requests at a running uvicorn worker, so the number
    includes the socket, the HTTP parser and the event loop.

Each mode measures three things, because a serial p95 on its own says nothing
about behaviour under load:

* serial: one request at a time; the latency a single caller sees.
* concurrent: N requests in flight at once, at several values of N,
  reporting both the latency distribution and sustained throughput. This is
  where a shared-model bug or a lock contention problem shows up.
* batch: the ``/predict/batch`` endpoint, per call and per project.

Results are written to ``reports/latency_<mode>_<segment>.json``. Payloads are
sampled from the held-out set so the request distribution matches reality
rather than one hand-picked example.

Implementation note: ``httpx.ASGITransport`` is async-only (it implements
``AsyncBaseTransport`` and has no ``__enter__``), so the in-process driver
runs on ``asyncio`` with ``httpx.AsyncClient``. Pairing it with the synchronous
``httpx.Client`` raises ``AttributeError: 'ASGITransport' object has no
attribute '__enter__'``.
See https://www.python-httpx.org/advanced/transports/

Run::

    python -m msa.bench.latency --requests 2000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import platform
import statistics
import time
from datetime import date

import numpy as np

from msa.config import DEFAULT_SEGMENT, REPORTS_DIR, SEGMENTS

LOG = logging.getLogger(__name__)

#: Concurrency levels measured by default. 1 is the serial number; the rest
#: answer "does it still hold up when several callers arrive at once".
DEFAULT_CONCURRENCY = (1, 8, 32)


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


# ---------------------------------------------------------------------------
# in-process (ASGI) driver
# ---------------------------------------------------------------------------


async def _drive_async(client, path: str, payloads: list[dict], concurrency: int):
    """Fire ``payloads`` at ``path`` with at most ``concurrency`` in flight."""
    samples: list[float] = []
    failures: list[int] = []
    semaphore = asyncio.Semaphore(concurrency)

    async def one(body: dict) -> None:
        async with semaphore:
            t0 = time.perf_counter()
            resp = await client.post(path, json=body)
            samples.append((time.perf_counter() - t0) * 1000.0)
            if resp.status_code != 200:
                failures.append(resp.status_code)

    wall0 = time.perf_counter()
    await asyncio.gather(*(one(p) for p in payloads))
    wall = time.perf_counter() - wall0
    if failures:
        raise RuntimeError(
            f"{len(failures)} of {len(payloads)} requests failed "
            f"(statuses: {sorted(set(failures))})"
        )
    return samples, wall


async def _bench_inprocess_async(
    segment: str,
    payloads: list[dict],
    warmup: int,
    concurrency_levels: tuple[int, ...],
    batch_size: int,
) -> dict:
    import httpx

    from msa.serve.app import create_app

    app = create_app(segment)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://bench", timeout=60.0
    ) as client:
        for p in payloads[:warmup]:
            await client.post("/predict", json=p)

        by_concurrency: dict[str, dict] = {}
        for level in concurrency_levels:
            samples, wall = await _drive_async(client, "/predict", payloads, level)
            stats = percentiles(samples)
            stats["concurrency"] = level
            stats["wall_seconds"] = wall
            stats["throughput_rps"] = len(payloads) / wall if wall else float("nan")
            by_concurrency[str(level)] = stats
            LOG.info(
                "concurrency %-3d p50=%.2f ms p95=%.2f ms  %.0f req/s",
                level,
                stats["p50_ms"],
                stats["p95_ms"],
                stats["throughput_rps"],
            )

        batches = [
            payloads[i : i + batch_size]
            for i in range(0, len(payloads) - batch_size, batch_size)
        ] or [payloads[:batch_size]]
        await client.post("/predict/batch", json={"items": batches[0]})
        batch_samples: list[float] = []
        for batch in batches:
            t0 = time.perf_counter()
            resp = await client.post("/predict/batch", json={"items": batch})
            batch_samples.append((time.perf_counter() - t0) * 1000.0)
            if resp.status_code != 200:
                raise RuntimeError(f"batch request failed: {resp.status_code}")
        batch_stats = percentiles(batch_samples)
        batch_stats["batch_size"] = batch_size
        batch_stats["per_item_p50_ms"] = batch_stats["p50_ms"] / batch_size

    return {"by_concurrency": by_concurrency, "batch": batch_stats}


def bench_inprocess(
    segment: str,
    payloads: list[dict],
    warmup: int,
    concurrency_levels: tuple[int, ...] = DEFAULT_CONCURRENCY,
    batch_size: int = 100,
) -> dict:
    return asyncio.run(
        _bench_inprocess_async(
            segment, payloads, warmup, concurrency_levels, batch_size
        )
    )


# ---------------------------------------------------------------------------
# real HTTP driver
# ---------------------------------------------------------------------------


def bench_http(
    url: str,
    payloads: list[dict],
    warmup: int,
    concurrency_levels: tuple[int, ...] = DEFAULT_CONCURRENCY,
) -> dict:
    """Drive a running uvicorn worker over a real socket."""
    import concurrent.futures

    import httpx

    with httpx.Client(base_url=url, timeout=60.0) as client:
        ready = client.get("/ready")
        if ready.status_code != 200:
            raise RuntimeError(
                f"service at {url} is not ready ({ready.status_code}); "
                f"it will not serve predictions"
            )
        for p in payloads[:warmup]:
            client.post("/predict", json=p)

        def _make_worker(samples: list[float], failures: list[int]):
            # bound explicitly rather than closing over the loop variables
            def one(body: dict) -> None:
                t0 = time.perf_counter()
                resp = client.post("/predict", json=body)
                samples.append((time.perf_counter() - t0) * 1000.0)
                if resp.status_code != 200:
                    failures.append(resp.status_code)

            return one

        by_concurrency: dict[str, dict] = {}
        for level in concurrency_levels:
            samples: list[float] = []
            failures: list[int] = []
            one = _make_worker(samples, failures)

            wall0 = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=level) as pool:
                list(pool.map(one, payloads))
            wall = time.perf_counter() - wall0
            if failures:
                raise RuntimeError(f"{len(failures)} requests failed: {sorted(set(failures))}")
            stats = percentiles(samples)
            stats["concurrency"] = level
            stats["wall_seconds"] = wall
            stats["throughput_rps"] = len(payloads) / wall if wall else float("nan")
            by_concurrency[str(level)] = stats
    return {"by_concurrency": by_concurrency, "batch": None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment", default=DEFAULT_SEGMENT, choices=list(SEGMENTS))
    ap.add_argument("--requests", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--mode", choices=["inprocess", "http"], default="inprocess")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument(
        "--concurrency",
        type=int,
        nargs="*",
        default=list(DEFAULT_CONCURRENCY),
        help="concurrency levels to measure (1 is the serial number)",
    )
    ap.add_argument("--batch-size", type=int, default=100)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    payloads = sample_payloads(args.segment, args.requests)
    levels = tuple(sorted(set(args.concurrency)))
    LOG.info(
        "benchmarking %d requests (%s) at concurrency %s",
        len(payloads),
        args.mode,
        list(levels),
    )

    if args.mode == "inprocess":
        measured = bench_inprocess(
            args.segment, payloads, args.warmup, levels, args.batch_size
        )
    else:
        measured = bench_http(args.url, payloads, args.warmup, levels)

    by_concurrency = measured["by_concurrency"]
    serial = by_concurrency.get("1") or next(iter(by_concurrency.values()))

    result = {
        "segment": args.segment,
        "mode": args.mode,
        "measured_on": str(date.today()),
        "host": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or "n/a",
            "cpu_count": None,
        },
        "single_request": serial,
        "by_concurrency": by_concurrency,
        "batch_request": measured["batch"],
    }
    import os

    result["host"]["cpu_count"] = os.cpu_count()

    out = REPORTS_DIR / f"latency_{args.mode}_{args.segment}.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    LOG.info(
        "serial p50=%.2f ms  p95=%.2f ms  p99=%.2f ms  -> %s",
        serial["p50_ms"],
        serial["p95_ms"],
        serial["p99_ms"],
        out,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
