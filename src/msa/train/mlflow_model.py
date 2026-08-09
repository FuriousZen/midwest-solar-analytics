"""A servable MLflow flavour for the timeline model.

Why this exists
---------------
``mlflow.xgboost.log_model`` logs the booster faithfully, but the resulting
model cannot be *served*. The booster was fitted with
``enable_categorical=True`` on pandas ``category`` columns; MLflow's serving
contract carries a DataFrame as JSON (``dataframe_split``), and JSON has no
categorical dtype. What comes back is ``object``, and XGBoost refuses it::

    ValueError: DataFrame.dtypes for data must be int, float, bool or category.
    ... Invalid columns: state: object, jurisdiction: object, source: object, ...

MLflow catches that during ``log_model`` and only *warns* ("Failed to validate
serving input example"), so the run looks fine and the registered model is
quietly unusable. This wrapper closes that gap: it restores the exact
categorical levels the booster was fitted on before predicting, and returns
days rather than raw booster output, so what the registry serves is the same
quantity the API serves.

The class deliberately imports nothing from ``msa`` at inference time -- its
whole configuration travels with it in ``metadata.json`` -- so a consumer can
load it with ``mlflow.pyfunc.load_model`` without this repository installed.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

#: Artifact keys inside the logged model.
BOOSTER_ARTIFACT = "booster"
METADATA_ARTIFACT = "metadata"


class TimelineModel(mlflow.pyfunc.PythonModel):
    """Predicts timeline **days** from a plain, JSON-round-trippable frame."""

    def load_context(self, context):  # pragma: no cover - exercised via pyfunc
        import xgboost as xgb

        self._meta = json.loads(
            Path(context.artifacts[METADATA_ARTIFACT]).read_text()
        )
        self._features: list[str] = list(self._meta["features"])
        self._categories: dict[str, list[str]] = self._meta["categories"]
        self._log_target: bool = bool(self._meta.get("log1p_target"))
        self._horizon: float = float(self._meta["horizon_days"])
        self._model = xgb.XGBRegressor()
        self._model.load_model(context.artifacts[BOOSTER_ARTIFACT])

    def _as_model_frame(self, model_input) -> pd.DataFrame:
        frame = pd.DataFrame(model_input).copy()
        missing = [c for c in self._features if c not in frame.columns]
        if missing:
            raise ValueError(f"input is missing required columns: {missing[:10]}")
        frame = frame[self._features]
        for name, levels in self._categories.items():
            values = frame[name].astype("object").where(frame[name].notna(), None)
            frame[name] = pd.Categorical(
                [None if v is None else str(v) for v in values], categories=levels
            )
        for name in self._features:
            if name not in self._categories:
                frame[name] = pd.to_numeric(frame[name], errors="coerce")
        return frame

    def predict(self, context, model_input, params=None):
        X = self._as_model_frame(model_input)
        raw = np.asarray(self._model.predict(X), dtype=float)
        if self._log_target:
            # expm1 of a large score overflows to +inf; clip in log space first,
            # exactly as msa.features.build.back_transform does.
            raw = np.expm1(np.clip(raw, 0.0, np.log1p(self._horizon)))
        return np.clip(raw, 0.0, self._horizon)


def json_safe_example(X: pd.DataFrame) -> pd.DataFrame:
    """A DataFrame that survives MLflow's JSON serving round trip.

    Categorical columns become strings and missing stays missing, which is
    exactly what a serving request looks like -- so the example MLflow
    validates is the shape a real caller sends.
    """
    out = X.copy()
    for name in out.columns:
        if isinstance(out[name].dtype, pd.CategoricalDtype):
            out[name] = out[name].astype("object").where(out[name].notna(), None)
    return out


__all__ = [
    "BOOSTER_ARTIFACT",
    "METADATA_ARTIFACT",
    "TimelineModel",
    "json_safe_example",
]
