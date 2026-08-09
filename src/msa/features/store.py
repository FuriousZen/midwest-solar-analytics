"""A small feature store so the API can rebuild training-time context features.

Several of the strongest features are *historical context* rather than
properties of the request: how many permits this contractor has filed before,
how fast the jurisdiction's desk has been running lately, how mature the solar
market is in that ZIP. A request body cannot carry those, and recomputing them
would mean shipping the whole corpus into the service.

Instead ``build_store`` snapshots the last observed value of each context
feature per key at the end of training, and ``apply_store`` joins those values
onto an incoming request. The service is then a pure function of
(request, snapshot), which is what makes a sub-50 ms budget achievable.

Staleness is explicit: the snapshot is dated, and ``/health`` reports that date.
Refreshing means re-running the pipeline.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)

CONTRACTOR_FEATURES = (
    "contractor_prior_permits",
    "contractor_prior_mean_days",
    "contractor_prior_median_days",
)
JURISDICTION_FEATURES = (
    "jurisdiction_apps_prev_30d",
    "jurisdiction_apps_prev_90d",
    "jurisdiction_apps_prev_365d",
    "jurisdiction_open_permits",
    "jurisdiction_issued_prev_30d",
    "jurisdiction_recent_mean_days_60d",
    "jurisdiction_recent_median_days_60d",
    "jurisdiction_recent_p90_days_180d",
    "jurisdiction_recent_mean_days_365d",
)
ZIP_FEATURES = (
    "tts_zip_cum_installs",
    "tts_zip_prior_mean_kw",
    "tts_zip_prior_median_ppw",
    "tts_zip_prior_tpo_share",
)
STATE_FEATURES = ("tts_state_installs_prev_365d",)


def _last_by(
    X: pd.DataFrame, keys: pd.Series, cols: tuple[str, ...], order: pd.Series
) -> dict[str, dict[str, float]]:
    frame = X[list(cols)].copy()
    frame["_k"] = keys.to_numpy()
    frame["_o"] = order.to_numpy()
    frame = frame.dropna(subset=["_k"])
    frame = frame.sort_values("_o").groupby("_k", sort=False).last()
    frame = frame.drop(columns=["_o"], errors="ignore")
    return {
        str(k): {c: (None if pd.isna(v) else float(v)) for c, v in row.items()}
        for k, row in frame.iterrows()
    }


def build_store(X: pd.DataFrame, meta: pd.DataFrame, raw: pd.DataFrame) -> dict:
    """Snapshot the latest context-feature values observed during training."""
    order = meta["start_date"]
    raw = raw.set_index("record_id").reindex(meta["record_id"]).reset_index()

    store = {
        "as_of": str(pd.Timestamp(order.max()).date()),
        "n_rows": int(len(X)),
        "contractor": _last_by(X, raw["contractor"], CONTRACTOR_FEATURES, order),
        "jurisdiction": _last_by(
            X, meta["jurisdiction"], JURISDICTION_FEATURES, order
        ),
        "zip": _last_by(X, raw["zip_code"], ZIP_FEATURES, order),
        "state": _last_by(X, meta["state"], STATE_FEATURES, order),
    }
    # global fallbacks: the median of each context feature across training
    store["defaults"] = {
        c: (None if pd.isna(v) else float(v))
        for c, v in X[
            list(
                CONTRACTOR_FEATURES
                + JURISDICTION_FEATURES
                + ZIP_FEATURES
                + STATE_FEATURES
            )
        ]
        .median()
        .items()
    }
    LOG.info(
        "feature store: %d contractors, %d jurisdictions, %d zips, as of %s",
        len(store["contractor"]),
        len(store["jurisdiction"]),
        len(store["zip"]),
        store["as_of"],
    )
    return store


def save_store(store: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store))


def load_store(path: Path) -> dict:
    return json.loads(path.read_text())


def apply_store(rows: pd.DataFrame, store: dict) -> pd.DataFrame:
    """Attach context features to request rows, in place-ish (returns a copy).

    ``rows`` must already carry ``jurisdiction``/``state``/``contractor``/
    ``zip_code`` columns (any of which may be null).
    """
    out = rows.copy()
    defaults = store.get("defaults", {})

    def lookup(table: str, key_col: str, cols: tuple[str, ...]) -> None:
        tbl = store.get(table, {})
        keys = out[key_col].astype("string") if key_col in out else pd.Series(
            [None] * len(out)
        )
        for c in cols:
            fallback = defaults.get(c)
            out[c] = [
                (tbl.get(str(k), {}) or {}).get(c, fallback)
                if k is not None and not pd.isna(k)
                else fallback
                for k in keys
            ]

    lookup("contractor", "contractor", CONTRACTOR_FEATURES)
    lookup("jurisdiction", "jurisdiction", JURISDICTION_FEATURES)
    lookup("zip", "zip_code", ZIP_FEATURES)
    lookup("state", "state", STATE_FEATURES)

    known = set(store.get("contractor", {}))
    out["contractor_is_known"] = [
        1 if (c is not None and not pd.isna(c) and str(c) in known) else 0
        for c in out.get("contractor", pd.Series([None] * len(out)))
    ]
    # Known, deliberate train/serve gap: the gap since a contractor's previous
    # filing needs their filing history relative to *this* request date, which
    # the snapshot cannot answer. It is left missing rather than approximated
    # with a stale value, so the booster routes these rows down its missing
    # branch instead of being fed a number that means something different from
    # what it meant in training. Documented in docs/METHODOLOGY.md.
    out["contractor_days_since_last_app"] = np.nan
    return out
