"""Ingest solar PV permits from the City of Chicago building-permit register.

Provenance
----------
Source      : City of Chicago Open Data Portal (Socrata)
Dataset     : "Building Permits", id ``ydr8-5enu``
URL         : https://data.cityofchicago.org/Buildings/Building-Permits/ydr8-5enu
Licence     : City of Chicago Open Data Portal Terms of Use (public domain data)
Timeline    : ``application_start_date`` -> ``issue_date``

Chicago issues a dedicated ``work_type`` of ``Small-Scale Solar PV System`` from
2023 onwards. Earlier solar jobs were filed under the Easy Permit Process with
no work_type, so we additionally match ``work_description`` on solar/PV keywords
and then exclude non-PV false positives (solariums, solar water heating,
solar-control window film).
"""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

from msa.config import CANONICAL_COLUMNS, RAW_DIR
from msa.ingest import socrata

LOG = logging.getLogger(__name__)

DOMAIN = "data.cityofchicago.org"
DATASET_ID = "ydr8-5enu"
SOURCE_KEY = "chicago_permits"

SOLAR_WORK_TYPE = "Small-Scale Solar PV System"

# SoQL predicate. NB: Socrata's `like` is case-sensitive, hence upper().
WHERE = (
    "(upper(work_description) like '%SOLAR%' "
    "or upper(work_description) like '%PHOTOVOLTAIC%' "
    f"or work_type = '{SOLAR_WORK_TYPE}') "
    "and application_start_date is not null and issue_date is not null"
)

_BASE_COLS = [
    "id",
    "permit_",
    "permit_status",
    "permit_milestone",
    "permit_type",
    "review_type",
    "application_start_date",
    "issue_date",
    "processing_time",
    "work_type",
    "work_description",
    "reported_cost",
    "total_fee",
    "building_fee_subtotal",
    "zoning_fee_subtotal",
    "other_fee_subtotal",
    "community_area",
    "census_tract",
    "ward",
    "latitude",
    "longitude",
]
_CONTACT_COLS = [
    f"contact_{i}_{f}" for i in range(1, 16) for f in ("type", "name", "zipcode")
]
SELECT = ",".join(_BASE_COLS + _CONTACT_COLS)

# --- description parsing -----------------------------------------------------

_RE_ARRAY_KW = re.compile(r"ARRAY OUTPUT:\s*([\d.]+)\s*KW")
_RE_INVERTER_KW = re.compile(r"INVERTER OUTPUT:\s*([\d.]+)\s*KW")
_RE_ANY_KW = re.compile(r"([\d.]+)\s*KW")
_RE_ROOF_TYPE = re.compile(r"ROOF TYPE:\s*([A-Z\-]+)")
_RE_BATTERY = re.compile(r"BATTER|ENERGY STORAGE|\bESS\b")
_RE_GROUND = re.compile(r"GROUND[- ]MOUNT|GROUND MOUNTED")

# Phrases that mean "not a PV installation" even though 'SOLAR' appears.
_NON_PV = re.compile(
    r"SOLARIUM"
    r"|SOLAR WATER HEAT|SOLAR HOT WATER|SOLAR THERMAL"
    r"|SOLAR TUBE|SOLATUBE|SOLAR SCREEN|SOLAR SHADE|SOLAR FILM|SOLAR CONTROL"
)
# Phrases that confirm it *is* PV.
_IS_PV = re.compile(r"SOLAR PANEL|PHOTOVOLTAIC|\bPV\b|SOLAR ARRAY|SOLAR SYSTEM|SOLAR ENERGY")


def _parse_float(series: pd.Series, pattern: re.Pattern[str]) -> pd.Series:
    return pd.to_numeric(series.str.extract(pattern, expand=False), errors="coerce")


def _first_contact_of(df: pd.DataFrame, wanted: tuple[str, ...]) -> pd.Series:
    """Return the name of the first contact slot whose type matches ``wanted``."""
    out = pd.Series(np.nan, index=df.index, dtype=object)
    for i in range(1, 16):
        tcol, ncol = f"contact_{i}_type", f"contact_{i}_name"
        if tcol not in df.columns:
            continue
        t = df[tcol].fillna("").str.upper()
        hit = t.isin(wanted) & out.isna()
        out.loc[hit] = df.loc[hit, ncol]
    return out


def _first_owner_zip(df: pd.DataFrame) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype=object)
    for i in range(1, 16):
        tcol, zcol = f"contact_{i}_type", f"contact_{i}_zipcode"
        if tcol not in df.columns:
            continue
        t = df[tcol].fillna("").str.upper()
        hit = (t == "OWNER") & out.isna()
        out.loc[hit] = df.loc[hit, zcol]
    return out


def normalize_contractor(name: object) -> object:
    """Upper-case and strip common corporate suffixes/punctuation."""
    if not isinstance(name, str) or not name.strip():
        return np.nan
    s = re.sub(r"[^A-Z0-9 ]+", " ", name.upper())
    s = re.sub(
        r"\b(LLC|L L C|INC|INCORPORATED|CO|CORP|CORPORATION|LTD|LP|LLP|COMPANY)\b",
        " ",
        s,
    )
    s = re.sub(r"\s+", " ", s).strip()
    return s or np.nan


def is_pv_description(desc: pd.Series, work_type: pd.Series) -> pd.Series:
    """Boolean mask selecting genuine PV jobs."""
    d = desc.fillna("").str.upper()
    tagged = work_type.fillna("") == SOLAR_WORK_TYPE
    return (tagged | d.str.contains(_IS_PV)) & ~d.str.contains(_NON_PV)


def fetch_raw(refresh: bool = False) -> pd.DataFrame:
    return socrata.fetch_csv(
        DOMAIN,
        DATASET_ID,
        select=SELECT,
        where=WHERE,
        order="id",
        cache_path=RAW_DIR / "chicago_building_permits_solar.csv",
        refresh=refresh,
    )


def to_canonical(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["work_description"] = df["work_description"].astype("string")
    df["work_type"] = df["work_type"].astype("string")

    keep = is_pv_description(df["work_description"], df["work_type"])
    LOG.info("Chicago: %d/%d rows kept as genuine PV", int(keep.sum()), len(df))
    df = df.loc[keep].copy()

    desc = df["work_description"].fillna("").str.upper()

    start = pd.to_datetime(df["application_start_date"], errors="coerce")
    end = pd.to_datetime(df["issue_date"], errors="coerce")

    array_kw = _parse_float(desc, _RE_ARRAY_KW)
    inv_kw = _parse_float(desc, _RE_INVERTER_KW)
    any_kw = _parse_float(desc, _RE_ANY_KW)
    size = array_kw.fillna(inv_kw).fillna(any_kw)
    # Sanity window for a small-scale (<=1 MW) distributed system.
    size = size.where((size > 0.2) & (size <= 1000))

    roof = df["work_description"].fillna("").str.upper().str.extract(
        _RE_ROOF_TYPE, expand=False
    )
    roof_type = pd.Series("unknown", index=df.index, dtype=object)
    roof_type[roof.eq("LOW")] = "flat"
    roof_type[roof.eq("SLOPED")] = "pitched"
    roof_type[roof.eq("STEEP")] = "steep"

    mount = np.where(desc.str.contains(_RE_GROUND), "ground", "roof")

    gc = _first_contact_of(
        df, ("GENERAL CONTRACTOR", "CONTRACTOR-GENERAL CONTRACTOR")
    )
    ec = _first_contact_of(df, ("ELECTRICAL CONTRACTOR", "CONTRACTOR-ELECTRICAL"))
    contractor = gc.fillna(ec).map(normalize_contractor)

    out = pd.DataFrame(
        {
            "record_id": SOURCE_KEY + ":" + df["id"].astype(str),
            "source": SOURCE_KEY,
            "jurisdiction": "Chicago",
            "state": "IL",
            "start_date": start,
            "end_date": end,
            "timeline_days": (end - start).dt.days,
            "system_size_kw": size,
            "reported_cost_usd": pd.to_numeric(df["reported_cost"], errors="coerce"),
            "permit_fee_usd": pd.to_numeric(df["total_fee"], errors="coerce"),
            "sector": np.where(
                df["permit_type"].fillna("").str.contains("NEW CONSTRUCTION"),
                "new_construction",
                "existing_building",
            ),
            "review_track": df["review_type"].fillna("UNKNOWN").str.title(),
            "contractor": contractor,
            "zip_code": _first_owner_zip(df).astype("string").str.slice(0, 5),
            "latitude": pd.to_numeric(df["latitude"], errors="coerce"),
            "longitude": pd.to_numeric(df["longitude"], errors="coerce"),
            "mount_type": mount,
            "roof_type": roof_type,
            "battery": desc.str.contains(_RE_BATTERY).astype(int),
        }
    )
    return out[list(CANONICAL_COLUMNS)]


def load(refresh: bool = False) -> pd.DataFrame:
    return to_canonical(fetch_raw(refresh=refresh))


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    frame = load()
    print(frame.shape)
    print(frame.head())
    print(frame["timeline_days"].describe())
