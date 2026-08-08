"""Ingest solar PV permits from Golden Valley, MN (PIMS permit system).

Provenance
----------
Source   : City of Golden Valley GIS -- "Golden_Misc" ArcGIS service,
           PIMS permit layers (586 = closed permits, 587 = open permits).
Layer    : https://gis.goldenvalleymn.gov/server/rest/services/GoldenMap/
           Golden_Misc/FeatureServer/586
Timeline : ``Applied_Date`` -> ``Issued_Date`` -> ``Final_Date``.

Golden Valley is small (~22k residents) but is the only Midwest register found
that publishes **all three** milestone dates, so it anchors both segments and
lets ``application_to_completion`` be measured end to end.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from msa.config import CANONICAL_COLUMNS, RAW_DIR
from msa.ingest import arcgis
from msa.ingest.text import (
    NON_PV_RE,
    PV_RE,
    extract_kw,
    has_battery,
    is_ground_mount,
    normalize_contractor,
)

LOG = logging.getLogger(__name__)

SOURCE_KEY = "goldenvalley_permits"
BASE = (
    "https://gis.goldenvalleymn.gov/server/rest/services/GoldenMap/"
    "Golden_Misc/FeatureServer"
)
LAYERS = (f"{BASE}/586", f"{BASE}/587")
WHERE = (
    "UPPER(Sub_Type) LIKE '%SOLAR%' OR UPPER(Work_Type) LIKE '%SOLAR%' "
    "OR UPPER(Description) LIKE '%SOLAR%' OR UPPER(Description) LIKE '%PHOTOVOLT%'"
)
OUT_FIELDS = ",".join(
    [
        "Permit_Nbr",
        "Permit_Type",
        "Sub_Type",
        "Work_Type",
        "Description",
        "Contractor",
        "Valuation",
        "totalFee",
        "Applied_Date",
        "Issued_Date",
        "Final_Date",
    ]
)
DATE_FIELDS = ("Applied_Date", "Issued_Date", "Final_Date")


def fetch_raw(refresh: bool = False) -> pd.DataFrame:
    frames = []
    for i, layer in enumerate(LAYERS):
        frames.append(
            arcgis.query(
                layer,
                where=WHERE,
                out_fields=OUT_FIELDS,
                order_by="Permit_Nbr",
                date_fields=DATE_FIELDS,
                cache_path=RAW_DIR / f"goldenvalley_pims_solar_{i}.csv",
                refresh=refresh,
            )
        )
    return pd.concat(frames, ignore_index=True)


def to_canonical(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["Description"] = df["Description"].astype("string")
    blob = (
        df["Description"].fillna("")
        + " "
        + df["Sub_Type"].astype("string").fillna("")
        + " "
        + df["Work_Type"].astype("string").fillna("")
    ).str.upper()

    keep = blob.str.contains(PV_RE) & ~blob.str.contains(NON_PV_RE)
    LOG.info("Golden Valley: %d/%d rows kept as PV", int(keep.sum()), len(df))
    df = df.loc[keep].drop_duplicates(subset=["Permit_Nbr"], keep="first").copy()

    text = df["Description"].fillna("").str.upper()
    size = extract_kw(text).where(lambda s: (s > 0.2) & (s <= 1000))

    out = pd.DataFrame(
        {
            "record_id": SOURCE_KEY + ":" + df["Permit_Nbr"].astype(str),
            "source": SOURCE_KEY,
            "jurisdiction": "Golden Valley",
            "state": "MN",
            "application_date": pd.to_datetime(df["Applied_Date"], errors="coerce"),
            "issue_date": pd.to_datetime(df["Issued_Date"], errors="coerce"),
            "completion_date": pd.to_datetime(df["Final_Date"], errors="coerce"),
            "system_size_kw": size,
            "reported_cost_usd": pd.to_numeric(df["Valuation"], errors="coerce"),
            "permit_fee_usd": pd.to_numeric(df["totalFee"], errors="coerce"),
            "sector": "unknown",
            "review_track": df["Permit_Type"].astype("string").fillna("Unknown"),
            "contractor": df["Contractor"].map(normalize_contractor),
            "zip_code": pd.NA,
            "latitude": np.nan,
            "longitude": np.nan,
            "mount_type": np.where(is_ground_mount(text), "ground", "roof"),
            "roof_type": "unknown",
            "battery": has_battery(text).astype(int),
            "permit_class": df["Sub_Type"].astype("string"),
            "district": pd.NA,
            "building_fee_usd": np.nan,
            "zoning_fee_usd": np.nan,
            "other_fee_usd": np.nan,
            "n_parties": np.nan,
            "has_expeditor": np.nan,
            "has_engineer": np.nan,
            "has_design_professional": np.nan,
            "description_length": df["Description"].str.len().astype("Float64"),
        }
    )
    return out[list(CANONICAL_COLUMNS)]


def load(refresh: bool = False) -> pd.DataFrame:
    return to_canonical(fetch_raw(refresh=refresh))


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    f = load()
    print(f.shape)
    print("applied->issued", (f.issue_date - f.application_date).dt.days.describe())
    print("issued->final", (f.completion_date - f.issue_date).dt.days.describe())
