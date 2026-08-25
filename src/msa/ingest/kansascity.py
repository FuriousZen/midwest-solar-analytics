"""Ingest solar PV permits from Kansas City, Missouri.

Provenance
----------
Source A : data.kcmo.org (Socrata), "Permits - CPD Dataset", id ``ntw8-aacc``
           Dates ``applieddate`` -> ``issueddate`` -> ``completeddate``.
           Deep history (1990s -> 2025-05) with contractor names, but the feed
           stopped updating in mid-2025.
Source B : data.kcmo.org (Socrata), "City Issued Permits", id ``w8jz-wjgn``
           Dates ``application_date`` -> ``issue_date`` -> ``finaled_date``.
           Live daily feed, but no contractor and no cost breakdown.

The two registers describe the same permit system in different eras and
overlap, so records are de-duplicated on the permit number with the
contractor-bearing CPD extract winning ties.

Solar is not a coded work class in either register; it is matched on the
free-text scope of work with the shared non-PV veto list applied (KCMO has a
noticeable number of *solarium* permits).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from msa.config import CANONICAL_COLUMNS, RAW_DIR
from msa.ingest import socrata
from msa.ingest.text import (
    NON_PV_RE,
    PV_RE,
    extract_kw,
    has_battery,
    is_ground_mount,
    normalize_contractor,
)

LOG = logging.getLogger(__name__)

SOURCE_KEY = "kansascity_permits"
DOMAIN = "data.kcmo.org"

CPD_ID = "ntw8-aacc"
CPD_WHERE = (
    "upper(description) like '%SOLAR%' or upper(description) like '%PHOTOVOLT%' "
    "or upper(workclass) like '%SOLAR%'"
)
CPD_SELECT = ",".join(
    [
        "permitnum",
        "applieddate",
        "issueddate",
        "completeddate",
        "description",
        "estprojectcost",
        "contractorcompanyname",
        "permittype",
        "permitclass",
        "permitclassmapped",
        "workclass",
        "statuscurrent",
        "originalzip",
        "latitude",
        "longitude",
    ]
)

CITY_ID = "w8jz-wjgn"
CITY_WHERE = "upper(project) like '%SOLAR%' or upper(work_class) like '%SOLAR%'"
CITY_SELECT = ",".join(
    [
        "permit_number",
        "application_date",
        "issue_date",
        "finaled_date",
        "project",
        "valuation",
        "permit_type",
        "work_class",
        "permit_status",
        "district",
        "main_zone",
    ]
)


def _fetch_cpd(refresh: bool) -> pd.DataFrame:
    return socrata.fetch_csv(
        DOMAIN,
        CPD_ID,
        select=CPD_SELECT,
        where=CPD_WHERE,
        order="permitnum",
        cache_path=RAW_DIR / "kcmo_permits_cpd_solar.csv",
        refresh=refresh,
    )


def _fetch_city(refresh: bool) -> pd.DataFrame:
    return socrata.fetch_csv(
        DOMAIN,
        CITY_ID,
        select=CITY_SELECT,
        where=CITY_WHERE,
        order="permit_number",
        cache_path=RAW_DIR / "kcmo_city_issued_permits_solar.csv",
        refresh=refresh,
    )


def _cpd_to_canonical(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["description"] = df["description"].astype("string")
    text = df["description"].fillna("").str.upper()
    keep = text.str.contains(PV_RE) & ~text.str.contains(NON_PV_RE)
    LOG.info("KCMO CPD: %d/%d rows kept as PV", int(keep.sum()), len(df))
    df = df.loc[keep].drop_duplicates(subset=["permitnum"], keep="first").copy()
    text = df["description"].fillna("").str.upper()

    cls = df["permitclassmapped"].astype("string").fillna("").str.upper()
    sector = np.where(
        cls.str.contains("NON-RES"),
        "commercial",
        np.where(cls.str.contains("RESIDENTIAL"), "residential", "unknown"),
    )
    return pd.DataFrame(
        {
            "record_id": SOURCE_KEY + ":" + df["permitnum"].astype(str),
            "source": SOURCE_KEY,
            "jurisdiction": "Kansas City",
            "state": "MO",
            "application_date": pd.to_datetime(df["applieddate"], errors="coerce"),
            "issue_date": pd.to_datetime(df["issueddate"], errors="coerce"),
            "completion_date": pd.to_datetime(df["completeddate"], errors="coerce"),
            "system_size_kw": extract_kw(text).where(lambda s: (s > 0.2) & (s <= 1000)),
            "reported_cost_usd": pd.to_numeric(df["estprojectcost"], errors="coerce"),
            "permit_fee_usd": np.nan,
            "sector": sector,
            "review_track": df["permittype"].astype("string").fillna("Unknown"),
            "contractor": df["contractorcompanyname"]
            .replace("NULL", np.nan)
            .replace("Needs Contact", np.nan)
            .map(normalize_contractor),
            "zip_code": df["originalzip"].astype("string").str.slice(0, 5),
            "latitude": pd.to_numeric(df["latitude"], errors="coerce"),
            "longitude": pd.to_numeric(df["longitude"], errors="coerce"),
            "mount_type": np.where(is_ground_mount(text), "ground", "roof"),
            "roof_type": "unknown",
            "battery": has_battery(text).astype(int),
            "permit_class": df["workclass"].astype("string"),
            "district": pd.NA,
            "building_fee_usd": np.nan,
            "zoning_fee_usd": np.nan,
            "other_fee_usd": np.nan,
            "n_parties": np.nan,
            "has_expeditor": np.nan,
            "has_engineer": np.nan,
            "has_design_professional": np.nan,
            "description_length": df["description"].str.len().astype("Float64"),
        }
    )[list(CANONICAL_COLUMNS)]


def _city_to_canonical(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["project"] = df["project"].astype("string")
    text = df["project"].fillna("").str.upper()
    keep = text.str.contains(PV_RE) & ~text.str.contains(NON_PV_RE)
    LOG.info("KCMO City: %d/%d rows kept as PV", int(keep.sum()), len(df))
    df = df.loc[keep].drop_duplicates(subset=["permit_number"], keep="first").copy()
    text = df["project"].fillna("").str.upper()

    ptype = df["permit_type"].astype("string").fillna("").str.upper()
    sector = np.where(
        ptype.str.contains("COMMERCIAL"),
        "commercial",
        np.where(ptype.str.contains("RESIDENTIAL"), "residential", "unknown"),
    )
    return pd.DataFrame(
        {
            "record_id": SOURCE_KEY + ":" + df["permit_number"].astype(str),
            "source": SOURCE_KEY,
            "jurisdiction": "Kansas City",
            "state": "MO",
            "application_date": pd.to_datetime(df["application_date"], errors="coerce"),
            "issue_date": pd.to_datetime(df["issue_date"], errors="coerce"),
            "completion_date": pd.to_datetime(df["finaled_date"], errors="coerce"),
            "system_size_kw": extract_kw(text).where(lambda s: (s > 0.2) & (s <= 1000)),
            "reported_cost_usd": pd.to_numeric(df["valuation"], errors="coerce"),
            "permit_fee_usd": np.nan,
            "sector": sector,
            "review_track": df["permit_type"].astype("string").fillna("Unknown"),
            "contractor": pd.NA,
            "zip_code": pd.NA,
            "latitude": np.nan,
            "longitude": np.nan,
            "mount_type": np.where(is_ground_mount(text), "ground", "roof"),
            "roof_type": "unknown",
            "battery": has_battery(text).astype(int),
            "permit_class": df["work_class"].astype("string"),
            "district": "district-" + df["district"].astype("string"),
            "building_fee_usd": np.nan,
            "zoning_fee_usd": np.nan,
            "other_fee_usd": np.nan,
            "n_parties": np.nan,
            "has_expeditor": np.nan,
            "has_engineer": np.nan,
            "has_design_professional": np.nan,
            "description_length": df["project"].str.len().astype("Float64"),
        }
    )[list(CANONICAL_COLUMNS)]


def load(refresh: bool = False) -> pd.DataFrame:
    cpd = _cpd_to_canonical(_fetch_cpd(refresh))
    city = _city_to_canonical(_fetch_city(refresh))
    # CPD first: it carries contractor names, which the live feed lacks.
    out = pd.concat([cpd, city], ignore_index=True)
    before = len(out)
    out = out.drop_duplicates(subset=["record_id"], keep="first")
    LOG.info(
        "KCMO: %d + %d rows -> %d after cross-register de-dup",
        len(cpd),
        len(city),
        len(out),
    )
    assert before >= len(out)
    return out.reset_index(drop=True)


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    f = load()
    print(f.shape)
    print("a->i", (f.issue_date - f.application_date).dt.days.describe())
    print("i->c", (f.completion_date - f.issue_date).dt.days.describe())
