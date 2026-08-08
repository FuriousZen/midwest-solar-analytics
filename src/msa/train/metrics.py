"""Metric helpers shared by the baseline, the Optuna study and the final model."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred, prefix: str = "") -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    err = np.abs(y_true - y_pred)
    out = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "median_ae": float(np.median(err)),
        "p90_ae": float(np.percentile(err, 90)),
        "within_3_days": float(np.mean(err <= 3)),
        "within_7_days": float(np.mean(err <= 7)),
        "bias": float(np.mean(y_pred - y_true)),
    }
    if prefix:
        out = {f"{prefix}{k}": v for k, v in out.items()}
    return out


def improvement_pct(baseline: float, candidate: float) -> float:
    """Percentage reduction in an error metric (positive == candidate better)."""
    if baseline == 0:
        return float("nan")
    return 100.0 * (baseline - candidate) / baseline
