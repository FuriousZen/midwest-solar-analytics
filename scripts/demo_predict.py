"""Worked example: call the prediction service and show the explanation.

Runs the app in-process (no server needed)::

    python -m scripts.demo_predict

Or against a running uvicorn worker::

    python -m scripts.demo_predict --url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment", default=DEFAULT_SEGMENT, choices=list(SEGMENTS))
    ap.add_argument("--url", default=None, help="target a running server instead")
    args = ap.parse_args()

    import httpx

    if args.url:
        client = httpx.Client(base_url=args.url, timeout=30)
    else:
        from msa.serve.app import create_app

        client = httpx.Client(
            transport=httpx.ASGITransport(app=create_app(args.segment)),
            base_url="http://demo",
        )

    with client:
        health = client.get("/health").json()
        print("== /health ==")
        print(json.dumps(health, indent=2))
        print()

        for ex in EXAMPLES:
            r = client.post("/predict?explain=true", json=ex["body"])
            r.raise_for_status()
            out = r.json()
            print(f"== {ex['name']} ==")
            print(
                f"   predicted {out['predicted_days']:.1f} days "
                f"-> {out['predicted_completion_date']}"
            )
            print("   top drivers:")
            for a in out["attributions"][:6]:
                print(f"     {a['shap']:+8.2f}  {a['feature']} = {a['value']}")
            print()

        print("== /metrics ==")
        print(json.dumps(client.get("/metrics").json(), indent=2))


if __name__ == "__main__":
    main()
