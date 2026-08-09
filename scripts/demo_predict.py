"""Worked example: call the prediction service and show the explanation.

Runs the app in-process (no server needed)::

    python -m scripts.demo_predict

Or against a running uvicorn worker::

    python -m scripts.demo_predict --url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import asyncio
import json

from msa.config import DEFAULT_SEGMENT, SEGMENTS

EXAMPLES = [
    {
        "name": "Minneapolis residential rooftop, mid-size, known installer",
        "body": {
            "jurisdiction": "Minneapolis",
            "state": "MN",
            "start_date": "2025-04-15",
            "system_size_kw": 8.4,
            "reported_cost_usd": 24000,
            "sector": "residential",
            "contractor": "ALL ENERGY SOLAR",
            "mount_type": "roof",
            "battery": 0,
        },
    },
    {
        "name": "Kansas City commercial array with storage",
        "body": {
            "jurisdiction": "Kansas City",
            "state": "MO",
            "start_date": "2025-06-02",
            "system_size_kw": 120.0,
            "reported_cost_usd": 310000,
            "sector": "commercial",
            "mount_type": "roof",
            "battery": 1,
        },
    },
    {
        "name": "Sparse record: only the two required fields",
        "body": {"jurisdiction": "Saint Paul", "start_date": "2024-09-01"},
    },
]


async def _run(segment: str, url: str | None) -> None:
    import httpx

    # httpx.ASGITransport implements AsyncBaseTransport and is documented as
    # requiring AsyncClient (https://www.python-httpx.org/advanced/transports/).
    # Pairing it with the synchronous Client raises
    # "'ASGITransport' object has no attribute '__enter__'", which is what this
    # script used to do -- so the demo could never actually run in-process.
    if url:
        client = httpx.AsyncClient(base_url=url, timeout=30)
    else:
        from msa.serve.app import create_app

        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(segment)),
            base_url="http://demo",
            timeout=30,
        )

    async with client:
        for endpoint in ("/health", "/ready"):
            r = await client.get(endpoint)
            print(f"== {endpoint} (HTTP {r.status_code}) ==")
            print(json.dumps(r.json(), indent=2))
            print()

        for ex in EXAMPLES:
            r = await client.post("/predict?explain=true", json=ex["body"])
            r.raise_for_status()
            out = r.json()
            print(f"== {ex['name']} ==")
            print(
                f"   predicted {out['predicted_days']:.1f} days "
                f"(80% interval {out['p10_days']:.1f}-{out['p90_days']:.1f}) "
                f"-> {out['predicted_completion_date']}"
            )
            print(f"   model_version {out['model_version']}")
            print("   top drivers:")
            for a in out["attributions"][:6]:
                print(f"     {a['shap']:+8.2f}  {a['feature']} = {a['value']}")
            print()

        # the refusal path is part of the demo: a caller should see what a bad
        # request looks like, not just a good one
        bad = await client.post(
            "/predict", json={"jurisdiction": "Minneapolis", "start_date": "not-a-date"}
        )
        print(f"== a malformed request (HTTP {bad.status_code}) ==")
        print(json.dumps(bad.json(), indent=2))
        print()

        r = await client.get("/metrics")
        print("== /metrics ==")
        print(json.dumps(r.json(), indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment", default=DEFAULT_SEGMENT, choices=list(SEGMENTS))
    ap.add_argument("--url", default=None, help="target a running server instead")
    args = ap.parse_args()
    asyncio.run(_run(args.segment, args.url))


if __name__ == "__main__":
    main()
