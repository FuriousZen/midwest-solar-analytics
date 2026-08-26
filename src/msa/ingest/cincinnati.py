"""Ingest solar PV permits from the City of Cincinnati, via a two-dataset join.

Provenance
----------
Portal   : data.cincinnati-oh.gov (Socrata), data maintained by CAGIS.
Dataset A: "Cincinnati Building Permits (Combo) w/ Plan Review Comments",
           id ``thvx-5mem``, the only Cincinnati extract that carries the
           free-text scope of work (``desc_of_work``). Used to *identify* solar.
Dataset B: "Cincinnati Building Permits Timeline", id ``94uh-66xv``, the
           milestone dates (``applied_date`` -> ``issued_date`` ->
           ``completeddate``). Used to *date* the identified permits.
Join key : ``number_key``.

Why the join is necessary: the headline "Cincinnati Building Permits"
(``uhjb-xac9``) has clean dates but its ``description`` column is a 16-value
category label (``Alteration``, ``HVAC``, ...), not scope text; searching it
for "solar" returns zero rows. Full-text search on that dataset matches the
street "SOLAR VISTA PL" and contractor names, not the work. So solar has to be
identified in ``thvx-5mem`` and dated from ``94uh-66xv``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from msa.config import CANONICAL_COLUMNS, RAW_DIR
from msa.ingest import socrata
from msa.ingest.text import (
    extract_kw,
    has_battery,
    is_ground_mount,
    is_pv,
)

LOG = logging.getLogger(__name__)

SOURCE_KEY = "cincinnati_permits"
DOMAIN = "data.cincinnati-oh.gov"

SCOPE_ID = "thvx-5mem"
SCOPE_WHERE = (
    "upper(desc_of_work) like '%SOLAR%' or upper(desc_of_work) like '%PHOTOVOLT%'"
)
SCOPE_SELECT = "number_key,desc_of_work,comp_type,sub_type,code_book,dwelling_units"

TIMELINE_ID = "94uh-66xv"
TIMELINE_SELECT = ",".join(
    [
        "number_key",
        "applied_date",
        "issued_date",
        "approved_date",
        "completeddate",
        "coissueddate",
        "work_type",
        "code_book",
        "dwelling_category",
        "dwelling_count",
        "est_value",
        "comp_type",
        "sub_type",
        "use_group",
        "data_status",
        "status_class",
        "totalsqft",
        "qty_revisions",
        "total_payments",
        "days_applied_to_approved",
    ]
)


def fetch_scope(refresh: bool = False) -> pd.DataFrame:
    return socrata.fetch_csv(
        DOMAIN,
        SCOPE_ID,
        select=SCOPE_SELECT,
        where=SCOPE_WHERE,
        order="number_key",
        cache_path=RAW_DIR / "cincinnati_scope_solar.csv",
        refresh=refresh,
    )


def fetch_timeline(keys: list[str], refresh: bool = False) -> pd.DataFrame:
    """Fetch timeline rows for the identified permits, in chunks of keys."""
    cache = RAW_DIR / "cincinnati_timeline_solar.csv"
    if cache.exists() and not refresh:
        LOG.info("Using cached %s", cache)
        return pd.read_csv(cache, low_memory=False)

    frames = []
    chunk = 400
    for i in range(0, len(keys), chunk):
        batch = keys[i : i + chunk]
        quoted = ",".join("'" + k.replace("'", "''") + "'" for k in batch)
        frames.append(
            socrata.fetch_csv(
                DOMAIN,
                TIMELINE_ID,
                select=TIMELINE_SELECT,
                where=f"number_key in ({quoted})",
                order="number_key",
            )
        )
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False)
    LOG.info("Cached %d timeline rows -> %s", len(out), cache)
    return out


def load(refresh: bool = False) -> pd.DataFrame:
    scope = fetch_scope(refresh=refresh)
    scope["desc_of_work"] = scope["desc_of_work"].astype("string")
    text = scope["desc_of_work"].fillna("").str.upper()
    keep = is_pv(text)
    LOG.info("Cincinnati scope: %d/%d rows kept as PV", int(keep.sum()), len(scope))
    scope = scope.loc[keep].copy()
    scope["number_key"] = scope["number_key"].astype(str)
    # one scope row per permit; keep the longest description
    scope["_len"] = scope["desc_of_work"].str.len().fillna(0)
    scope = (
        scope.sort_values(["number_key", "_len"], ascending=[True, False])
        .drop_duplicates("number_key", keep="first")
        .drop(columns="_len")
    )

    tl = fetch_timeline(scope["number_key"].tolist(), refresh=refresh)
    tl["number_key"] = tl["number_key"].astype(str)
    tl = tl.drop_duplicates("number_key", keep="first")

    df = scope.merge(tl, on="number_key", how="inner", suffixes=("_scope", ""))
    LOG.info("Cincinnati: %d solar permits matched to timeline rows", len(df))
    if df.empty:
        return pd.DataFrame(columns=list(CANONICAL_COLUMNS))

    text = df["desc_of_work"].fillna("").str.upper()
    dwell = df["dwelling_category"].astype("string").fillna("").str.upper()
    sector = np.where(
        dwell.str.contains("SINGLE|TWO|MULTI|DWELL"),
        "residential",
        np.where(dwell.eq(""), "unknown", "commercial"),
    )

    out = pd.DataFrame(
        {
            "record_id": SOURCE_KEY + ":" + df["number_key"],
            "source": SOURCE_KEY,
            "jurisdiction": "Cincinnati",
            "state": "OH",
            "application_date": pd.to_datetime(df["applied_date"], errors="coerce"),
            "issue_date": pd.to_datetime(df["issued_date"], errors="coerce"),
            "completion_date": pd.to_datetime(df["completeddate"], errors="coerce"),
            "system_size_kw": extract_kw(text).where(lambda s: (s > 0.2) & (s <= 1000)),
            "reported_cost_usd": pd.to_numeric(df["est_value"], errors="coerce"),
            "permit_fee_usd": pd.to_numeric(df["total_payments"], errors="coerce"),
            "sector": sector,
            "review_track": df["comp_type"].astype("string").fillna("Unknown"),
            "contractor": pd.NA,
            "zip_code": pd.NA,
            "latitude": np.nan,
            "longitude": np.nan,
            "mount_type": np.where(is_ground_mount(text), "ground", "roof"),
            "roof_type": "unknown",
            "battery": has_battery(text).astype(int),
            "permit_class": df["sub_type"].astype("string"),
            "district": pd.NA,
            "building_fee_usd": np.nan,
            "zoning_fee_usd": np.nan,
            "other_fee_usd": np.nan,
            "n_parties": np.nan,
            "has_expeditor": np.nan,
            "has_engineer": np.nan,
            "has_design_professional": np.nan,
            "description_length": text.str.len().astype("Float64"),
        }
    )
    return out[list(CANONICAL_COLUMNS)]


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    f = load()
    print(f.shape)
    print("a->i", (f.issue_date - f.application_date).dt.days.describe())
    print("i->c", (f.completion_date - f.issue_date).dt.days.describe())
