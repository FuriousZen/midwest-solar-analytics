"""Ingest solar PV permits from the City of Saint Paul permit register.

Provenance
----------
Source   : City of Saint Paul Open Data (ArcGIS Hub) -- "Approved Building Permits"
Layer    : https://services1.arcgis.com/9meaaHE3uiba0zr8/arcgis/rest/services/
           Approved_Building_Permits/FeatureServer/0
Licence  : City of Saint Paul open data.
Timeline : ``ISSUEDATE`` -> ``FINALDATE`` (permit issued -> final inspection),
           i.e. the ``permit_to_completion`` segment.

Saint Paul is the best-labelled Midwest register found: an explicit
``SUB_TYPE`` of ``Solar PV Systems`` / ``Solar``. We use that code as the
primary filter and fall back to the description for the parallel building-permit
record of the same job.

Note: this feed stopped updating on 2025-06-30 (verified against the whole
layer, not just the solar subset), so it is a closed historical series.
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

SOURCE_KEY = "saintpaul_permits"
LAYER = (
    "https://services1.arcgis.com/9meaaHE3uiba0zr8/arcgis/rest/services/"
    "Approved_Building_Permits/FeatureServer/0"
)
WHERE = (
    "SUB_TYPE IN ('Solar','Solar PV Systems') "
    "OR UPPER(FOLDERDESCRIPTION) LIKE '%SOLAR%' "
    "OR UPPER(FOLDERDESCRIPTION) LIKE '%PHOTOVOLT%'"
)
OUT_FIELDS = ",".join(
    [
        "PERMITNUMBER",
        "FOLDER_TYPE",
        "SUB_TYPE",
        "WORK_TYPE",
        "STATUS",
        "FOLDERDESCRIPTION",
        "EST_VALUE_OF_WORK",
        "STATE_VALUATION",
        "CONTRACTORNAME",
        "ISSUEDATE",
        "FINALDATE",
        "ADDRESS",
        "WARD",
        "DISTRINCT",
        "LATITUE",
        "LONGITUTE",
    ]
)
DATE_FIELDS = ("ISSUEDATE", "FINALDATE")

_RE_DC_KW = re.compile(r"\bDC[\s:]*([\d.]+)")
SOLAR_SUB_TYPES = {"Solar", "Solar PV Systems"}


def fetch_raw(refresh: bool = False) -> pd.DataFrame:
    return arcgis.query(
        LAYER,
        where=WHERE,
        out_fields=OUT_FIELDS,
        order_by="PERMITNUMBER",
        date_fields=DATE_FIELDS,
        cache_path=RAW_DIR / "saintpaul_building_permits_solar.csv",
        refresh=refresh,
    )


def to_canonical(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["FOLDERDESCRIPTION"] = df["FOLDERDESCRIPTION"].astype("string")
    text = df["FOLDERDESCRIPTION"].fillna("").str.upper()
    sub = df["SUB_TYPE"].astype("string").fillna("")

    coded = sub.isin(SOLAR_SUB_TYPES)
    keep = (coded | text.str.contains(PV_RE)) & ~text.str.contains(NON_PV_RE)
    LOG.info(
        "Saint Paul: %d/%d rows kept (%d via SUB_TYPE code)",
        int(keep.sum()),
        len(df),
        int(coded.sum()),
    )
    df = df.loc[keep].copy()
    df = df.drop_duplicates(subset=["PERMITNUMBER"], keep="first")

    text = df["FOLDERDESCRIPTION"].fillna("").str.upper()
    size = pd.to_numeric(text.str.extract(_RE_DC_KW, expand=False), errors="coerce")
    size = size.fillna(extract_kw(text))
    size = size.where((size > 0.2) & (size <= 1000))

    work = df["WORK_TYPE"].astype("string").fillna("")
    sector = np.where(
        work.str.upper().str.contains("COMMERCIAL"),
        "commercial",
        np.where(work.str.upper().str.contains("RESIDENTIAL"), "residential", "unknown"),
    )

    out = pd.DataFrame(
        {
            "record_id": SOURCE_KEY + ":" + df["PERMITNUMBER"].astype(str),
            "source": SOURCE_KEY,
            "jurisdiction": "Saint Paul",
            "state": "MN",
            "application_date": pd.NaT,
            "issue_date": pd.to_datetime(df["ISSUEDATE"], errors="coerce"),
            "completion_date": pd.to_datetime(df["FINALDATE"], errors="coerce"),
            "system_size_kw": size,
            "reported_cost_usd": pd.to_numeric(
                df["EST_VALUE_OF_WORK"], errors="coerce"
            ),
            "permit_fee_usd": np.nan,
            "sector": sector,
            "review_track": df["FOLDER_TYPE"].astype("string").fillna("Unknown"),
            "contractor": df["CONTRACTORNAME"].map(normalize_contractor),
            "zip_code": pd.NA,
            "latitude": pd.to_numeric(df["LATITUE"], errors="coerce"),
            "longitude": pd.to_numeric(df["LONGITUTE"], errors="coerce"),
            "mount_type": np.where(is_ground_mount(text), "ground", "roof"),
            "roof_type": "unknown",
            "battery": has_battery(text).astype(int),
            "permit_class": df["SUB_TYPE"].astype("string"),
            "district": "district-" + df["DISTRINCT"].astype("string"),
            "building_fee_usd": np.nan,
            "zoning_fee_usd": np.nan,
            "other_fee_usd": np.nan,
            "n_parties": np.nan,
            "has_expeditor": np.nan,
            "has_engineer": np.nan,
            "has_design_professional": np.nan,
            "description_length": df["FOLDERDESCRIPTION"].str.len().astype("Float64"),
        }
    )
    return out[list(CANONICAL_COLUMNS)]


def load(refresh: bool = False) -> pd.DataFrame:
    return to_canonical(fetch_raw(refresh=refresh))


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    f = load()
    print(f.shape)
    print((f["completion_date"] - f["issue_date"]).dt.days.describe())
