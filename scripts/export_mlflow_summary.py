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

    runs.to_csv(CSV_PATH, index=False)

    # A tag column only exists once at least one run carried that tag. Falling
    # back to an empty Series would produce a length-0 object that blows up
    # crosstab and boolean indexing, killing the pipeline at this step; build a
    # correctly-sized default instead.
    def tag(name: str, default: str) -> pd.Series:
        if f"tags.{name}" in runs.columns:
            return runs[f"tags.{name}"].fillna(default).astype(str)
        return pd.Series([default] * len(runs), index=runs.index, dtype=str)

    stage = tag("stage", "untagged")
    segment = tag("segment", "n/a")

    by_stage = stage.value_counts().to_dict()
    by_segment_stage = (
        pd.crosstab(segment, stage).to_dict(orient="index") if len(runs) else {}
    )

    best_rows = []
    is_trial = stage == "optuna"
    if "metrics.cv_mae" in runs.columns and is_trial.any():
        trials = runs[is_trial].copy()
        for seg, sub in trials.groupby(segment[is_trial]):
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
