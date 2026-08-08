"""Ingest solar PV permits from the City of Minneapolis permit register.

Provenance
----------
Source   : City of Minneapolis Open Data (ArcGIS Hub) -- "CCS Permits"
Layer    : https://services.arcgis.com/afSMGVsC7QlRK1kZ/arcgis/rest/services/
           CCS_Permits/FeatureServer/0
Licence  : City of Minneapolis open data, no restrictions on reuse.
Timeline : ``issueDate`` -> ``completeDate``  (permit issued -> permit closed,
           i.e. the field installation + inspection window).

Minneapolis has no application date, so this source contributes only to the
``permit_to_completion`` segment.

Solar jobs are identifiable only from the free-text ``comments`` scope of work;
there is no solar work-type code. Rows repeat per permit (one row per
sub-record), so we de-duplicate on ``permitNumber``.
"""

from __future__ import annotations

import logging
import re

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

SOURCE_KEY = "minneapolis_permits"
LAYER = (
    "https://services.arcgis.com/afSMGVsC7QlRK1kZ/arcgis/rest/services/"
    "CCS_Permits/FeatureServer/0"
)
WHERE = "UPPER(comments) LIKE '%SOLAR%' OR UPPER(comments) LIKE '%PHOTOVOLT%'"
OUT_FIELDS = ",".join(
    [
        "permitNumber",
        "permitType",
        "workType",
        "occupancyType",
        "status",
        "milestone",
        "value",
        "totalFees",
        "comments",
        "issueDate",
        "completeDate",
        "applicantName",
        "fullName",
        "Neighborhoods_Desc",
        "Wards",
        "Display",
    ]
)
DATE_FIELDS = ("issueDate", "completeDate")

#: Minneapolis templates the array size as e.g. "COLLECTOR CAPACITY  6.16  KW".
_RE_COLLECTOR = re.compile(r"COLLECTOR CAPACITY[\s_]*([\d.]+)[\s_]*KW")

_COMMERCIAL = {"Comm", "Mixed", "MFD", "3to4"}


def fetch_raw(refresh: bool = False) -> pd.DataFrame:
    return arcgis.query(
        LAYER,
        where=WHERE,
        out_fields=OUT_FIELDS,
        order_by="permitNumber",
        date_fields=DATE_FIELDS,
        cache_path=RAW_DIR / "minneapolis_ccs_permits_solar.csv",
        refresh=refresh,
    )


def to_canonical(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["comments"] = df["comments"].astype("string")
    text = df["comments"].fillna("").str.upper()

    keep = text.str.contains(PV_RE) & ~text.str.contains(NON_PV_RE)
    LOG.info("Minneapolis: %d/%d rows match PV keywords", int(keep.sum()), len(df))
    df = df.loc[keep].copy()

    # one row per permit; keep the record with the most complete dates
    df["_score"] = df[["issueDate", "completeDate"]].notna().sum(axis=1)
    df = (
        df.sort_values(["permitNumber", "_score"], ascending=[True, False])
        .drop_duplicates(subset=["permitNumber"], keep="first")
        .drop(columns="_score")
    )
    LOG.info("Minneapolis: %d distinct permits", len(df))

    text = df["comments"].fillna("").str.upper()
    size = pd.to_numeric(text.str.extract(_RE_COLLECTOR, expand=False), errors="coerce")
    size = size.fillna(extract_kw(text))
    size = size.where((size > 0.2) & (size <= 1000))

    occ = df["occupancyType"].astype("string").fillna("")
    sector = np.where(occ.isin(_COMMERCIAL), "commercial", "residential")
    sector = np.where(occ.eq(""), "unknown", sector)

    out = pd.DataFrame(
        {
            "record_id": SOURCE_KEY + ":" + df["permitNumber"].astype(str),
            "source": SOURCE_KEY,
            "jurisdiction": "Minneapolis",
            "state": "MN",
            "application_date": pd.NaT,
            "issue_date": pd.to_datetime(df["issueDate"], errors="coerce"),
            "completion_date": pd.to_datetime(df["completeDate"], errors="coerce"),
            "system_size_kw": size,
            "reported_cost_usd": pd.to_numeric(df["value"], errors="coerce"),
            "permit_fee_usd": pd.to_numeric(df["totalFees"], errors="coerce"),
            "sector": sector,
            "review_track": df["permitType"].astype("string").fillna("Unknown"),
            "contractor": df["applicantName"].map(normalize_contractor),
            "zip_code": pd.NA,
            "latitude": np.nan,
            "longitude": np.nan,
            "mount_type": np.where(is_ground_mount(text), "ground", "roof"),
            "roof_type": "unknown",
            "battery": has_battery(text).astype(int),
            "permit_class": df["workType"].astype("string"),
            "district": "ward-" + df["Wards"].astype("string").str.replace(
                r"\.0$", "", regex=True
            ),
            "building_fee_usd": np.nan,
            "zoning_fee_usd": np.nan,
            "other_fee_usd": np.nan,
            "n_parties": np.nan,
            "has_expeditor": np.nan,
            "has_engineer": np.nan,
            "has_design_professional": np.nan,
            "description_length": df["comments"].str.len().astype("Float64"),
        }
    )
    return out[list(CANONICAL_COLUMNS)]


def load(refresh: bool = False) -> pd.DataFrame:
    return to_canonical(fetch_raw(refresh=refresh))


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    f = load()
    print(f.shape)
    d = (f["completion_date"] - f["issue_date"]).dt.days
    print(d.describe())
