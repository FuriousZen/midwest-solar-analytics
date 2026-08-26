"""Run the whole pipeline end to end, offline, in a scratch directory.

This is the fast, network-free equivalent of ``./scripts/run_pipeline.sh``:
it exercises every real entrypoint (baselines, Optuna, final fit, SHAP
dashboard, the FastAPI service, the latency benchmark and the MLflow export)
against a small synthetic corpus.

It exists so CI can prove the *chain* still works after every change, which is
a different question from whether the units work. A broken hand-off between
``final.py`` and ``serve/app.py`` passes every unit test and fails here.

Nothing it writes touches the repository: paths and the MLflow store are all
redirected into ``--root`` (a temp directory by default) via the environment,
before ``msa`` is imported.

It is not a measurement. The corpus is generated, so no number it prints
means anything about solar permits; the only assertion is that each stage
produces the artifact the next stage needs.

Run::

    python -m scripts.smoke_pipeline               # temp dir, 3 trials
    python -m scripts.smoke_pipeline --trials 5 --root /tmp/smoke --keep
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SEGMENTS = ("permit_to_completion", "application_to_issue")


def synthetic_corpus(n: int, seed: int = 11):
    """A canonical-schema corpus with enough structure to fit something.

    Deliberately generated, and labelled as such everywhere it is written, so
    it can never be mistaken for the real one. It only has to be *shaped* like
    the real corpus: three milestone dates, several jurisdictions, a contractor
    column with repeat filers, and durations that depend on the features.
    """
    import numpy as np
    import pandas as pd

    from msa.config import CANONICAL_COLUMNS

    rng = np.random.default_rng(seed)
    jurisdictions = ["Smokeville", "Testburg", "Fixtureton"]
    contractors = [f"SMOKE CONTRACTOR {i}" for i in range(12)]

    juris = rng.choice(jurisdictions, n)
    tracks = rng.choice(["Express", "Standard", "Complex"], n)
    size = rng.uniform(3, 40, n)
    battery = rng.integers(0, 2, n)

    application = pd.Timestamp("2018-01-01") + pd.to_timedelta(
        rng.integers(0, 365 * 5, n), unit="D"
    )
    # review and build times depend on the features, so a model has something
    # real to find and the metrics are not pure noise
    review = np.clip(
        4
        + 8 * (tracks == "Standard")
        + 20 * (tracks == "Complex")
        + rng.normal(0, 3, n),
        0,
        None,
    ).astype(int)
    build = np.clip(
        25 + 1.4 * size + 18 * battery + rng.normal(0, 12, n), 0, None
    ).astype(int)

    frame = pd.DataFrame(dict.fromkeys(CANONICAL_COLUMNS), index=range(n))
    frame["record_id"] = [f"smoke:{i}" for i in range(n)]
    frame["source"] = "smoke_generated"
    frame["jurisdiction"] = juris
    frame["state"] = np.where(juris == "Smokeville", "IL", "MN")
    frame["application_date"] = application
    frame["issue_date"] = application + pd.to_timedelta(review, unit="D")
    frame["completion_date"] = frame["issue_date"] + pd.to_timedelta(build, unit="D")
    frame["system_size_kw"] = size
    frame["reported_cost_usd"] = size * rng.uniform(2200, 3600, n)
    frame["permit_fee_usd"] = rng.uniform(50, 900, n)
    frame["building_fee_usd"] = rng.uniform(0, 500, n)
    frame["zoning_fee_usd"] = rng.choice([0.0, 50.0], n)
    frame["other_fee_usd"] = rng.uniform(0, 300, n)
    frame["sector"] = rng.choice(["residential", "commercial"], n)
    frame["review_track"] = tracks
    frame["contractor"] = rng.choice(contractors, n)
    frame["zip_code"] = rng.choice(["60601", "60602", "55401", "55402"], n)
    frame["latitude"] = rng.uniform(41, 45, n)
    frame["longitude"] = rng.uniform(-93, -87, n)
    frame["mount_type"] = rng.choice(["roof", "ground"], n)
    frame["roof_type"] = rng.choice(["flat", "pitched"], n)
    frame["battery"] = battery
    frame["permit_class"] = rng.choice(["Alteration", "New"], n)
    frame["district"] = rng.choice(["ward-1", "ward-2", "ward-3"], n)
    frame["n_parties"] = rng.integers(1, 6, n)
    frame["has_expeditor"] = rng.integers(0, 2, n)
    frame["has_engineer"] = rng.integers(0, 2, n)
    frame["has_design_professional"] = rng.integers(0, 2, n)
    frame["description_length"] = rng.integers(20, 400, n)
    return frame[list(CANONICAL_COLUMNS)]


class Runner:
    def __init__(self, env: dict[str, str], verbose: bool):
        self.env = env
        self.verbose = verbose
        self.timings: list[tuple[str, float]] = []

    def run(self, label: str, *args: str) -> None:
        print(f"\n==> {label}\n    $ python -m {' '.join(args)}", flush=True)
        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, "-m", *args],
            env=self.env,
            cwd=REPO_ROOT,
            capture_output=not self.verbose,
            text=True,
        )
        elapsed = time.perf_counter() - t0
        self.timings.append((label, elapsed))
        if proc.returncode != 0:
            if not self.verbose:
                sys.stdout.write(proc.stdout or "")
                sys.stderr.write(proc.stderr or "")
            raise SystemExit(f"FAILED after {elapsed:.1f}s: {label}")
        print(f"    ok ({elapsed:.1f}s)", flush=True)


def require(paths: list[Path], stage: str) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"{stage} did not produce: {missing}")


def check_service(models_dir: Path, segment: str) -> None:
    """Load the fitted model through the real app and make a real prediction."""
    from fastapi.testclient import TestClient

    from msa.serve.app import create_app

    app = create_app(segment, root=models_dir)
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200, health.text
        assert health.json()["model_loaded"] is True
        ready = client.get("/ready")
        assert ready.status_code == 200, ready.text

        jurisdiction = health.json()["jurisdictions"][0]
        body = {
            "jurisdiction": jurisdiction,
            "start_date": "2022-06-01",
            "state": "IL",
            "system_size_kw": 9.2,
        }
        single = client.post("/predict?explain=true", json=body)
        assert single.status_code == 200, single.text
        payload = single.json()
        assert 0 <= payload["predicted_days"] <= payload["horizon_days"]
        assert payload["attributions"], "explain=true returned no attributions"
        assert payload["model_version"] == health.json()["model_version"]

        batch = client.post("/predict/batch", json={"items": [body] * 25})
        assert batch.status_code == 200, batch.text
        assert len(batch.json()["predictions"]) == 25

        # and the failure path still refuses rather than guessing
        bad = client.post("/predict", json={"jurisdiction": jurisdiction})
        assert bad.status_code == 422, bad.text

    print(
        f"    served {segment}: {payload['predicted_days']} days "
        f"[{payload['p10_days']}, {payload['p90_days']}] "
        f"via {payload['model_version']}",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None, help="scratch directory (default: temp)")
    ap.add_argument("--trials", type=int, default=3, help="Optuna trials per segment")
    ap.add_argument("--rows", type=int, default=4000, help="synthetic records")
    ap.add_argument("--keep", action="store_true", help="do not delete the scratch dir")
    ap.add_argument("--verbose", action="store_true", help="stream subprocess output")
    args = ap.parse_args()

    root = Path(args.root) if args.root else Path(tempfile.mkdtemp(prefix="msa-smoke-"))
    root.mkdir(parents=True, exist_ok=True)
    data, models, reports = root / "data", root / "models", root / "reports"
    for d in (data / "processed", models, reports):
        d.mkdir(parents=True, exist_ok=True)

    env = {
        **os.environ,
        "MSA_DATA_DIR": str(data),
        "MSA_MODELS_DIR": str(models),
        "MSA_REPORTS_DIR": str(reports),
        "MSA_DISABLE_TTS": "1",
        "MLFLOW_TRACKING_URI": f"sqlite:///{root / 'mlflow.db'}",
        "MLFLOW_ARTIFACT_ROOT": str(root / "mlartifacts"),
        "MLFLOW_EXPERIMENT": "smoke",
        "PYTHONPATH": str(REPO_ROOT / "src"),
    }
    # the in-process checks below must see the same redirected paths
    os.environ.update(env)

    print(f"smoke pipeline in {root} ({args.trials} trials/segment)", flush=True)

    frame = synthetic_corpus(args.rows)
    frame.to_parquet(data / "processed" / "installations.parquet", index=False)
    (data / "processed" / "dataset_summary.json").write_text(
        json.dumps(
            {
                "GENERATED": "synthetic smoke corpus, not a measurement",
                "total_records": len(frame),
                "rows_with_usable_duration_by_segment": {
                    "permit_to_completion": len(frame),
                    "application_to_issue": len(frame),
                    "application_to_completion": len(frame),
                },
                "failed_sources": {},
                "sources": [],
            },
            indent=2,
        )
    )
    print(f"    wrote {len(frame):,} synthetic canonical records", flush=True)

    runner = Runner(env, args.verbose)
    try:
        runner.run("horizon coverage", "scripts.report_horizon_coverage")
        require([reports / "horizon_coverage.json"], "horizon coverage")

        for seg in SEGMENTS:
            runner.run(f"baselines [{seg}]", "msa.train.baseline", "--segment", seg)
            require([reports / seg / "baseline_results.json"], "baselines")

            runner.run(
                f"optuna x{args.trials} [{seg}]",
                "msa.train.tune",
                "--segment",
                seg,
                "--trials",
                str(args.trials),
                # in-memory storage: a smoke run must not touch the repo's
                # optuna.db, which holds the real study
                "--storage",
                "",
            )
            require(
                [reports / seg / "best_params.json", reports / seg / "optuna_study.json"],
                "optuna",
            )

            runner.run(f"final fit [{seg}]", "msa.train.final", "--segment", seg)
            require(
                [
                    models / seg / "model.json",
                    models / seg / "model_quantiles.json",
                    models / seg / "metadata.json",
                    models / seg / "feature_store.json",
                    reports / seg / "final_metrics.json",
                ],
                "final",
            )

            runner.run(f"shap dashboard [{seg}]", "msa.explain.shap_report", "--segment", seg)
            require([reports / seg / "shap_dashboard.html"], "explain")

            print(f"\n==> serve + predict [{seg}]", flush=True)
            check_service(models, seg)

            runner.run(
                f"latency benchmark [{seg}]",
                "msa.bench.latency",
                "--segment",
                seg,
                "--requests",
                "200",
                "--warmup",
                "20",
            )
            require([reports / f"latency_inprocess_{seg}.json"], "bench")

        runner.run("mlflow export", "scripts.export_mlflow_summary")
        require([reports / "mlflow_runs_summary.json"], "mlflow export")
    finally:
        total = sum(t for _, t in runner.timings)
        print("\n--- stage timings ---")
        for label, seconds in runner.timings:
            print(f"  {seconds:7.1f}s  {label}")
        print(f"  {total:7.1f}s  total")
        if args.keep:
            print(f"\nscratch kept at {root}")
        else:
            shutil.rmtree(root, ignore_errors=True)

    print("\nsmoke pipeline OK: every stage produced the artifact the next one needs.")


if __name__ == "__main__":
    main()
