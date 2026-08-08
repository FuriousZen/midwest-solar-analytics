"""Export a committable summary of the MLflow experiment.

The tracking store itself (``mlflow.db`` + ``mlartifacts/``) is gitignored
because it is large and machine-specific, so this writes the evidence that the
runs happened -- counts by stage, the full run table, and the top trials -- into
``reports/mlflow_runs_summary.json`` and ``reports/mlflow_runs.csv``, which are
committed.

Run::

    python -m scripts.export_mlflow_summary
"""

from __future__ import annotations

import json
import logging

import mlflow
import pandas as pd

from msa.config import MLFLOW_EXPERIMENT, REPORTS_DIR
from msa.train.tracking import init_mlflow

LOG = logging.getLogger(__name__)

CSV_PATH = REPORTS_DIR / "mlflow_runs.csv"
JSON_PATH = REPORTS_DIR / "mlflow_runs_summary.json"


def export() -> dict:
    init_mlflow()
    runs = mlflow.search_runs(
        experiment_names=[MLFLOW_EXPERIMENT],
        output_format="pandas",
        max_results=50_000,
    )
    if runs.empty:
        raise RuntimeError(
            "no MLflow runs found -- run the training pipeline before exporting"
        )

    keep = [c for c in runs.columns if not c.startswith("params.") or True]
    runs = runs[keep]
    runs.to_csv(CSV_PATH, index=False)

    stage = runs.get("tags.stage", pd.Series(dtype=str)).fillna("untagged")
    segment = runs.get("tags.segment", pd.Series(dtype=str)).fillna("n/a")

    by_stage = stage.value_counts().to_dict()
    by_segment_stage = (
        pd.crosstab(segment, stage).to_dict(orient="index") if len(runs) else {}
    )

    best_rows = []
    if "metrics.cv_mae" in runs.columns:
        trials = runs[stage == "optuna"].copy()
        for seg, sub in trials.groupby(segment[stage == "optuna"]):
            top = sub.nsmallest(5, "metrics.cv_mae")
            for _, r in top.iterrows():
                best_rows.append(
                    {
                        "segment": seg,
                        "run_name": r.get("tags.mlflow.runName"),
                        "cv_mae": float(r["metrics.cv_mae"]),
                        "cv_r2": float(r.get("metrics.cv_r2", float("nan"))),
                        "cv_rmse": float(r.get("metrics.cv_rmse", float("nan"))),
                    }
                )

    summary = {
        "experiment": MLFLOW_EXPERIMENT,
        "total_runs": int(len(runs)),
        "runs_by_stage": {str(k): int(v) for k, v in by_stage.items()},
        "runs_by_segment_and_stage": {
            str(k): {str(kk): int(vv) for kk, vv in v.items()}
            for k, v in by_segment_stage.items()
        },
        "first_run_started": str(runs["start_time"].min()),
        "last_run_started": str(runs["start_time"].max()),
        "top_trials_by_cv_mae": best_rows,
        "run_table_csv": str(CSV_PATH.relative_to(REPORTS_DIR.parent)),
    }
    JSON_PATH.write_text(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(json.dumps(export(), indent=2, default=str))
