"""MLflow bootstrap shared by every training entrypoint."""

from __future__ import annotations

import logging

import mlflow

from msa.config import (
    MLFLOW_ARTIFACT_ROOT,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
)

LOG = logging.getLogger(__name__)


def init_mlflow(experiment: str = MLFLOW_EXPERIMENT) -> str:
    """Point MLflow at the local file store and return the experiment id."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    if client.get_experiment_by_name(experiment) is None:
        client.create_experiment(
            experiment, artifact_location=f"file://{MLFLOW_ARTIFACT_ROOT}"
        )
    exp = mlflow.set_experiment(experiment)
    LOG.info(
        "MLflow tracking_uri=%s experiment=%s (id=%s)",
        MLFLOW_TRACKING_URI,
        experiment,
        exp.experiment_id,
    )
    return exp.experiment_id


def count_runs(experiment: str = MLFLOW_EXPERIMENT) -> int:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    runs = mlflow.search_runs(experiment_names=[experiment], output_format="pandas")
    return int(len(runs))
