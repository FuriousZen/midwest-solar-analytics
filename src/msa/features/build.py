"""Feature engineering for the installation-timeline regression.

Design rule: **every feature must be computable from information available on
the day the application is filed** (``start_date``). Nothing may depend on
``end_date`` of the record itself or of any other record, because ``end_date``
is exactly what the model predicts and would not be known at serving time for
in-flight permits.

Three families of features are produced:

1. ``application``  -- attributes of the job itself (size, cost, review track,
   calendar seasonality, geography).
2. ``contractor``   -- strictly-backward-looking history for the contractor on
   the permit (how many jobs they have filed before, how long those took).
   Implemented as an expanding mean over *prior* records only, so it is a
   causal statistic rather than target leakage.
3. ``market``       -- jurisdiction workload in the trailing 30/90 days, plus
   utility-record context from LBNL Tracking the Sun as of the application
   date (cumulative installs / median system size / median $ per watt in the
   applicant's ZIP and state).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from msa.config import (
    DEFAULT_SEGMENT,
    MIN_TIMELINE_DAYS,
    SEGMENTS,
    TARGET,
    max_timeline_days,
)

LOG = logging.getLogger(__name__)

#: Columns fed to the model. Order is fixed so that the serving payload,
#: the SHAP report and the trained booster always agree.
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "state",
    "jurisdiction",
    "source",
    "sector",
    "review_track",
    "mount_type",
    "roof_type",
    "permit_class",
    "district",
    "timeline_segment",
)

# NOTE: absolute time indices (``start_year``, days-since-series-start) are
# deliberately EXCLUDED. They let the booster memorise the level of each
# historical period, which inflates random-fold CV but collapses on a
# chronological holdout because the value is out of range for future permits.
# Measured effect on the chronological test set: R2 -0.10 -> +0.17 for stock
# XGBoost when they are removed (see reports/ablation_time_features.json).
NUMERIC_FEATURES: tuple[str, ...] = (
    "system_size_kw",
    "log_system_size_kw",
    "reported_cost_usd",
    "log_reported_cost_usd",
    "cost_per_kw",
    "permit_fee_usd",
    "building_fee_usd",
    "zoning_fee_usd",
    "other_fee_usd",
    "zoning_review_required",
    "battery",
    "n_parties",
    "has_expeditor",
    "has_engineer",
    "has_design_professional",
    "description_length",
    "start_month",
    "start_dayofweek",
    "start_dayofyear",
    "start_quarter",
    "start_is_month_end",
    "start_week_of_year",
    "latitude",
    "longitude",
    "contractor_prior_permits",
    "contractor_prior_mean_days",
    "contractor_prior_median_days",
    "contractor_is_known",
    "contractor_days_since_last_app",
    "jurisdiction_apps_prev_30d",
    "jurisdiction_apps_prev_90d",
    "jurisdiction_apps_prev_365d",
    "jurisdiction_open_permits",
    "jurisdiction_issued_prev_30d",
    "jurisdiction_recent_mean_days_60d",
    "jurisdiction_recent_median_days_60d",
    "jurisdiction_recent_p90_days_180d",
    "jurisdiction_recent_mean_days_365d",
    "tts_zip_cum_installs",
    "tts_zip_prior_mean_kw",
    "tts_zip_prior_median_ppw",
    "tts_zip_prior_tpo_share",
    "tts_state_installs_prev_365d",
)

FEATURES: tuple[str, ...] = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------


def _one_segment(
    df: pd.DataFrame, segment: str, apply_censor_guard: bool
) -> pd.DataFrame:
    start_col, end_col = SEGMENTS[segment]
    horizon = max_timeline_days(segment)

    out = df.copy()
    for c in ("application_date", "issue_date", "completion_date"):
        out[c] = pd.to_datetime(out[c], errors="coerce")
    out = out.dropna(subset=[start_col, end_col])
    if out.empty:
        return out.assign(
            timeline_segment=segment, start_date=pd.NaT, end_date=pd.NaT, **{TARGET: []}
        )

    out["timeline_segment"] = segment
    out["start_date"] = out[start_col]
    out["end_date"] = out[end_col]
    out[TARGET] = (out["end_date"] - out["start_date"]).dt.days
    n_dates = len(out)

    out = out[out[TARGET].between(MIN_TIMELINE_DAYS, horizon, inclusive="both")]
    n_horizon = len(out)

    if apply_censor_guard and not out.empty:
        # Right-censoring is per-source: each portal has its own snapshot date.
        keep = pd.Series(True, index=out.index)
        for src, sub in out.groupby("source", sort=False):
            cutoff = sub["end_date"].max() - pd.Timedelta(days=horizon)
            keep.loc[sub.index] = sub["start_date"] <= cutoff
        out = out[keep]

    out["record_id"] = out["record_id"].astype(str) + "|" + segment
    out = out.drop_duplicates(subset=["record_id"])
    LOG.info(
        "segment %-26s %6d rows with both dates -> %6d within %d-day horizon "
        "-> %6d after censor guard",
        segment,
        n_dates,
        n_horizon,
        horizon,
        len(out),
    )
    return out


def clean(
    df: pd.DataFrame,
    segment: str = DEFAULT_SEGMENT,
    apply_censor_guard: bool = True,
) -> pd.DataFrame:
    """Turn canonical milestone dates into a modelling frame for one segment.

    ``segment='pooled'`` stacks every segment a source can support and records
    which milestones each row spans in ``timeline_segment``. Rows are
    de-duplicated on ``record_id|segment``, so a source that publishes all three
    dates legitimately contributes to more than one segment.
    """
    n0 = len(df)
    segments = (
        list(SEGMENTS) if segment == "pooled" else [segment]
    )
    frames = [
        f
        for f in (_one_segment(df, s, apply_censor_guard) for s in segments)
        if not f.empty
    ]
    if not frames:
        raise ValueError(f"no rows available for segment(s) {segments}")
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["start_date", "record_id"]).reset_index(drop=True)
    LOG.info("clean(segment=%s): %d canonical rows -> %d modelling rows", segment, n0, len(out))
    return out


# ---------------------------------------------------------------------------
# Feature blocks
# ---------------------------------------------------------------------------

#: Queue statistics are per (jurisdiction, milestone pair): a permit desk's
#: review backlog and its field-inspection backlog are different queues.
_QUEUE_KEY = ["jurisdiction", "timeline_segment"]


def _application_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df["start_date"]
    out = pd.DataFrame(index=df.index)
    size = pd.to_numeric(df["system_size_kw"], errors="coerce")
    cost = pd.to_numeric(df["reported_cost_usd"], errors="coerce")
    cost = cost.where(cost > 0)
    out["system_size_kw"] = size
    out["log_system_size_kw"] = np.log1p(size)
    out["reported_cost_usd"] = cost
    out["log_reported_cost_usd"] = np.log1p(cost)
    out["cost_per_kw"] = (cost / size).replace([np.inf, -np.inf], np.nan)
    for c in (
        "permit_fee_usd",
        "building_fee_usd",
        "zoning_fee_usd",
        "other_fee_usd",
        "battery",
        "n_parties",
        "has_expeditor",
        "has_engineer",
        "has_design_professional",
        "description_length",
    ):
        out[c] = pd.to_numeric(df[c], errors="coerce") if c in df else np.nan
    out["zoning_review_required"] = (out["zoning_fee_usd"] > 0).astype(float)
    out.loc[out["zoning_fee_usd"].isna(), "zoning_review_required"] = np.nan
    out["start_month"] = d.dt.month
    out["start_dayofweek"] = d.dt.dayofweek
    out["start_dayofyear"] = d.dt.dayofyear
    out["start_quarter"] = d.dt.quarter
    out["start_is_month_end"] = d.dt.is_month_end.astype(int)
    out["start_week_of_year"] = d.dt.isocalendar().week.astype(int)
    out["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    return out


def _contractor_features(df: pd.DataFrame) -> pd.DataFrame:
    """Expanding, strictly-prior contractor statistics.

    ``df`` must already be sorted by ``start_date``. ``shift(1)`` guarantees the
    current record never contributes to its own feature value.
    """
    out = pd.DataFrame(index=df.index)
    key = df["contractor"].fillna("__UNKNOWN__")
    known = key.ne("__UNKNOWN__")

    grp = df.assign(_k=key).groupby("_k", sort=False)[TARGET]
    prior_n = grp.cumcount()
    prior_mean = grp.apply(lambda s: s.shift(1).expanding().mean()).reset_index(
        level=0, drop=True
    )
    prior_median = grp.apply(lambda s: s.shift(1).expanding().median()).reset_index(
        level=0, drop=True
    )

    out["contractor_prior_permits"] = np.where(known, prior_n, np.nan)
    out["contractor_prior_mean_days"] = np.where(known, prior_mean, np.nan)
    out["contractor_prior_median_days"] = np.where(known, prior_median, np.nan)
    out["contractor_is_known"] = known.astype(int)

    gap = (
        df.assign(_k=key)
        .groupby("_k", sort=False)["start_date"]
        .diff()
        .dt.days
    )
    out["contractor_days_since_last_app"] = np.where(known, gap, np.nan)
    return out


def _queue_features(df: pd.DataFrame) -> pd.DataFrame:
    """Reviewer-queue state observable on the application date.

    * ``jurisdiction_open_permits`` -- applications already filed but not yet
      issued as of ``start_date``. Whether a *previous* permit is still open is
      known on day ``t``; only its eventual issue date is unknown, and that is
      not used.
    * ``jurisdiction_issued_prev_30d`` -- desk throughput over the past month.
    * ``jurisdiction_recent_*_days_*`` -- the realised turnaround of permits
      *already issued* before ``start_date``. Those durations are fully
      observed at time ``t``, so this is the single most informative
      leak-free signal about the desk's current speed.
    """
    cols = [
        "jurisdiction_open_permits",
        "jurisdiction_issued_prev_30d",
        "jurisdiction_recent_mean_days_60d",
        "jurisdiction_recent_median_days_60d",
        "jurisdiction_recent_p90_days_180d",
        "jurisdiction_recent_mean_days_365d",
    ]
    out = pd.DataFrame(np.nan, index=df.index, columns=cols, dtype=float)

    for _, idx in df.groupby(_QUEUE_KEY, sort=False).groups.items():
        sub = df.loc[idx, ["start_date", "end_date", TARGET]]
        t = sub["start_date"].to_numpy("datetime64[ns]")
        starts = np.sort(sub["start_date"].to_numpy("datetime64[ns]"))
        ends = np.sort(sub["end_date"].to_numpy("datetime64[ns]"))

        started_before = np.searchsorted(starts, t, side="left")
        finished_by = np.searchsorted(ends, t, side="right")
        out.loc[idx, "jurisdiction_open_permits"] = (
            started_before - finished_by
        ).astype(float)

        window30 = t - np.timedelta64(30, "D")
        out.loc[idx, "jurisdiction_issued_prev_30d"] = (
            np.searchsorted(ends, t, side="left")
            - np.searchsorted(ends, window30, side="left")
        ).astype(float)

        # rolling statistics over permits keyed by their ISSUE date
        by_issue = sub.sort_values("end_date")
        series = pd.Series(
            by_issue[TARGET].to_numpy(dtype=float),
            index=pd.DatetimeIndex(by_issue["end_date"]),
        )
        rolled = pd.DataFrame(
            {
                "jurisdiction_recent_mean_days_60d": series.rolling("60D").mean(),
                "jurisdiction_recent_median_days_60d": series.rolling("60D").median(),
                "jurisdiction_recent_p90_days_180d": series.rolling("180D").quantile(
                    0.9
                ),
                "jurisdiction_recent_mean_days_365d": series.rolling("365D").mean(),
            }
        )
        rolled = rolled.groupby(level=0).last().sort_index()
        rolled["end_date"] = rolled.index

        left = df.loc[idx, ["start_date"]].copy()
        left["_row"] = np.arange(len(left))
        left = left.sort_values("start_date")
        merged = pd.merge_asof(
            left,
            rolled.reset_index(drop=True).sort_values("end_date"),
            left_on="start_date",
            right_on="end_date",
            direction="backward",
            allow_exact_matches=False,
        ).sort_values("_row")
        for c in (
            "jurisdiction_recent_mean_days_60d",
            "jurisdiction_recent_median_days_60d",
            "jurisdiction_recent_p90_days_180d",
            "jurisdiction_recent_mean_days_365d",
        ):
            out.loc[idx, c] = merged[c].to_numpy()
    return out


def _rolling_application_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Number of other applications filed in the same jurisdiction recently.

    A proxy for reviewer workload/backlog. Uses only ``start_date``, which is
    observable at prediction time for every record involved.
    """
    out = pd.DataFrame(index=df.index, dtype=float)
    for window, name in ((30, "30d"), (90, "90d"), (365, "365d")):
        col = f"jurisdiction_apps_prev_{name}"
        out[col] = np.nan
        for _, idx in df.groupby(_QUEUE_KEY, sort=False).groups.items():
            sub = df.loc[idx, ["start_date"]].copy()
            ones = pd.Series(1.0, index=pd.DatetimeIndex(sub["start_date"]))
            # inclusive rolling count minus self => strictly-prior count
            rolled = ones.rolling(f"{window}D").sum().to_numpy() - 1.0
            out.loc[idx, col] = rolled
    return out


def _tts_context(df: pd.DataFrame, tts: pd.DataFrame | None) -> pd.DataFrame:
    """As-of joins against LBNL Tracking the Sun utility records.

    All statistics are computed over TTS systems installed *strictly before*
    each permit's application date, so no future information leaks in. Where
    the permit post-dates the end of TTS coverage the statistic saturates at
    its last observed value, which is the correct "market maturity to date"
    reading.
    """
    cols = [
        "tts_zip_cum_installs",
        "tts_zip_prior_mean_kw",
        "tts_zip_prior_median_ppw",
        "tts_zip_prior_tpo_share",
        "tts_state_installs_prev_365d",
    ]
    out = pd.DataFrame(np.nan, index=df.index, columns=cols, dtype=float)
    if tts is None or tts.empty:
        return out

    t = tts.copy()
    t = t[t["installation_date"].between("2000-01-01", "2100-01-01")]
    t["zip_code"] = t["zip_code"].astype("string")
    size = pd.to_numeric(t["pv_system_size_dc"], errors="coerce")
    price = pd.to_numeric(t["total_installed_price"], errors="coerce")
    t["ppw"] = (price / (size * 1000.0)).replace([np.inf, -np.inf], np.nan)
    t["ppw"] = t["ppw"].where(t["ppw"].between(0.5, 25))
    t["tpo"] = pd.to_numeric(t["third_party_owned"], errors="coerce").clip(0, 1)
    t = t.sort_values("installation_date")

    # ---- ZIP-level expanding statistics -----------------------------------
    g = t.groupby("zip_code", sort=False)
    zip_frame = pd.DataFrame(
        {
            "zip_code": t["zip_code"].to_numpy(),
            "installation_date": t["installation_date"].to_numpy(),
            "tts_zip_cum_installs": g.cumcount().to_numpy() + 1.0,
            "tts_zip_prior_mean_kw": g["pv_system_size_dc"]
            .transform(lambda s: s.expanding().mean())
            .to_numpy(),
            "tts_zip_prior_median_ppw": g["ppw"]
            .transform(lambda s: s.expanding().median())
            .to_numpy(),
            "tts_zip_prior_tpo_share": g["tpo"]
            .transform(lambda s: s.expanding().mean())
            .to_numpy(),
        }
    )
    zip_frame["zip_code"] = zip_frame["zip_code"].astype("string")
    zip_frame = zip_frame.sort_values(["installation_date", "zip_code"])

    left = df[["start_date"]].copy()
    left["zip_code"] = df["zip_code"].astype("string")
    left["_row"] = np.arange(len(left))
    left = left.sort_values("start_date")

    merged = pd.merge_asof(
        left,
        zip_frame,
        left_on="start_date",
        right_on="installation_date",
        by="zip_code",
        direction="backward",
        allow_exact_matches=False,
    ).sort_values("_row")

    for c in (
        "tts_zip_cum_installs",
        "tts_zip_prior_mean_kw",
        "tts_zip_prior_median_ppw",
        "tts_zip_prior_tpo_share",
    ):
        out[c] = merged[c].to_numpy()

    # ---- state-level trailing 365-day install volume -----------------------
    daily = (
        t.groupby(["state", t["installation_date"].dt.normalize()])
        .size()
        .rename("n")
        .reset_index()
        .rename(columns={"installation_date": "day"})
    )
    pieces = []
    for state, sub in daily.groupby("state", sort=False):
        s = sub.set_index("day")["n"].asfreq("D", fill_value=0)
        roll = s.rolling("365D").sum()
        pieces.append(
            pd.DataFrame(
                {
                    "state": state,
                    "day": roll.index,
                    "tts_state_installs_prev_365d": roll.to_numpy(),
                }
            )
        )
    state_frame = pd.concat(pieces, ignore_index=True).sort_values(["day", "state"])

    left2 = df[["start_date", "state"]].copy()
    left2["_row"] = np.arange(len(left2))
    left2 = left2.sort_values("start_date")
    merged2 = pd.merge_asof(
        left2,
        state_frame,
        left_on="start_date",
        right_on="day",
        by="state",
        direction="backward",
        allow_exact_matches=False,
    ).sort_values("_row")
    out["tts_state_installs_prev_365d"] = merged2[
        "tts_state_installs_prev_365d"
    ].to_numpy()
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_features(
    df: pd.DataFrame, tts: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(X, y)`` ready for XGBoost with native categorical support."""
    df = df.sort_values(["start_date", "record_id"]).reset_index(drop=True)
    blocks = [
        _application_features(df),
        _contractor_features(df),
        _rolling_application_counts(df),
        _queue_features(df),
        _tts_context(df, tts),
    ]
    X = pd.concat(blocks, axis=1)
    for c in CATEGORICAL_FEATURES:
        X[c] = df[c].astype("string").fillna("unknown").astype("category")
    X = X[list(FEATURES)]
    y = df[TARGET].astype(float)
    return X, y


def as_model_frame(records: pd.DataFrame, categories: dict[str, list[str]]) -> pd.DataFrame:
    """Coerce an arbitrary feature frame to the exact training dtypes/order.

    Used by the serving layer, where categorical levels must match the levels
    the booster was trained on or XGBoost will map them to different codes.
    """
    out = records.copy()
    for c in FEATURES:
        if c not in out.columns:
            out[c] = np.nan
    for c in CATEGORICAL_FEATURES:
        out[c] = pd.Categorical(
            out[c].astype("string").fillna("unknown"), categories=categories[c]
        )
    for c in NUMERIC_FEATURES:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out[list(FEATURES)]
