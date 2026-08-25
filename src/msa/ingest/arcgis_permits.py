"""Declarative ingestion for the ArcGIS-hosted Midwest permit registers.

Detroit MI, Cleveland OH, Naperville IL, Overland Park KS, Ames IA and Bismarck
ND all publish the same shape of record (a permit number, a scope-of-work string
and two or three milestone dates) behind ArcGIS FeatureServer/MapServer query
endpoints. Rather than four near-identical modules, each is expressed as a
:class:`PermitLayer` spec and shares one canonicaliser.

Each spec was verified by fetching the layer's own ``?f=json`` field list before
being written down; the field names below are the layer's real field names.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

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


@dataclass(frozen=True)
class PermitLayer:
    source_key: str
    jurisdiction: str
    state: str
    layer_url: str
    where: str
    id_field: str
    description_field: str
    application_field: str | None
    issue_field: str | None
    completion_field: str | None
    cache_name: str
    cost_field: str | None = None
    fee_field: str | None = None
    contractor_field: str | None = None
    permit_type_field: str | None = None
    work_class_field: str | None = None
    zip_field: str | None = None
    lat_field: str | None = None
    lon_field: str | None = None
    district_field: str | None = None
    extra_fields: tuple[str, ...] = field(default_factory=tuple)

    @property
    def date_fields(self) -> tuple[str, ...]:
        return tuple(
            f
            for f in (self.application_field, self.issue_field, self.completion_field)
            if f
        )

    @property
    def out_fields(self) -> str:
        cols = [
            self.id_field,
            self.description_field,
            *self.date_fields,
            self.cost_field,
            self.fee_field,
            self.contractor_field,
            self.permit_type_field,
            self.work_class_field,
            self.zip_field,
            self.lat_field,
            self.lon_field,
            self.district_field,
            *self.extra_fields,
        ]
        seen: list[str] = []
        for c in cols:
            if c and c not in seen:
                seen.append(c)
        return ",".join(seen)


LAYERS: dict[str, PermitLayer] = {
    "detroit_permits": PermitLayer(
        source_key="detroit_permits",
        jurisdiction="Detroit",
        state="MI",
        layer_url=(
            "https://services2.arcgis.com/qvkbeam7Wirps6zC/arcgis/rest/services/"
            "bseed_building_permits/FeatureServer/0"
        ),
        where=(
            "UPPER(work_description) LIKE '%SOLAR%' "
            "OR UPPER(work_description) LIKE '%PHOTOVOLT%'"
        ),
        id_field="record_id",
        description_field="work_description",
        application_field="submitted_date",
        issue_field="issued_date",
        # BSEED publishes no completion date on the building-permit layer.
        completion_field=None,
        cost_field="amt_estimated_contractor_cost",
        fee_field="amt_permit_cost",
        permit_type_field="permit_type",
        work_class_field="current_use_type",
        zip_field="zip_code",
        lat_field="latitude",
        lon_field="longitude",
        district_field="council_district",
        cache_name="detroit_building_permits_solar.csv",
        extra_fields=("num_units", "zoning_designation", "neighborhood"),
    ),
    "cleveland_permits": PermitLayer(
        source_key="cleveland_permits",
        jurisdiction="Cleveland",
        state="OH",
        layer_url=(
            "https://services3.arcgis.com/dty2kHktVXHrqO8i/arcgis/rest/services/"
            "Building_Permits/FeatureServer/0"
        ),
        where=(
            "UPPER(WORK_DESCRIPTION) LIKE '%SOLAR%' "
            "OR UPPER(WORK_DESCRIPTION) LIKE '%PHOTOVOLT%'"
        ),
        id_field="PERMIT_ID",
        description_field="WORK_DESCRIPTION",
        application_field="FILE_DATE",
        issue_field="ISSUE_DATE",
        completion_field=None,
        cost_field="JOB_VALUE",
        contractor_field="CONTRATOR_BUSINESS_NAME",
        permit_type_field="PERMIT_TYPE",
        work_class_field="USE_GROUP_1",
        fee_field="TOTAL_FEES_PAID",
        lat_field="LAT",
        lon_field="LON",
        district_field="WARD",
        cache_name="cleveland_building_permits_solar.csv",
        extra_fields=("PERMIT_SUBTYPE", "PERMIT_SUBMITTED_ONLINE"),
    ),
    "naperville_permits": PermitLayer(
        source_key="naperville_permits",
        jurisdiction="Naperville",
        state="IL",
        layer_url=(
            "https://services1.arcgis.com/rXJ6QApc2sOtl1Pd/arcgis/rest/services/"
            "Building_Permits_View/FeatureServer/0"
        ),
        # Naperville is one of only two registers found with an explicit solar
        # work class, so no text matching is needed for recall.
        where=(
            "PERMITWORKCLASS = 'SOLAR/PHOTOVOLTAIC' "
            "OR UPPER(DESCRIPTION) LIKE '%SOLAR%' "
            "OR UPPER(DESCRIPTION) LIKE '%PHOTOVOLT%'"
        ),
        id_field="PERMITNUMBER",
        description_field="DESCRIPTION",
        application_field="APPLYDATE",
        issue_field="ISSUEDATE",
        completion_field="FINALIZEDATE",
        cost_field="PERMITVALUATION",
        permit_type_field="PERMITTYPE",
        work_class_field="PERMITWORKCLASS",
        zip_field="POSTALCODE",
        cache_name="naperville_building_permits_solar.csv",
        extra_fields=("PERMITSTATUS",),
    ),
    "overlandpark_permits": PermitLayer(
        source_key="overlandpark_permits",
        jurisdiction="Overland Park",
        state="KS",
        layer_url=(
            "https://services1.arcgis.com/YQsWDBr0DjMtoTQo/arcgis/rest/services/"
            "Building_Permits/FeatureServer/0"
        ),
        where=(
            "WorkClass = 'Solar Panel' "
            "OR UPPER(Description) LIKE '%SOLAR%' "
            "OR UPPER(Description) LIKE '%PHOTOVOLT%'"
        ),
        id_field="CaseNumber",
        description_field="Description",
        application_field="ApplyDate",
        issue_field="IssueDate",
        completion_field="FinalizedDate",
        cost_field="Valuation",
        fee_field="TotalFeeAmount",
        permit_type_field="PermitType",
        work_class_field="WorkClass",
        lat_field="Latitude",
        lon_field="Longitude",
        district_field="WARD",
        cache_name="overlandpark_building_permits_solar.csv",
    ),
    "ames_permits": PermitLayer(
        source_key="ames_permits",
        jurisdiction="Ames",
        state="IA",
        layer_url=(
            "https://gis.cityofames.org/arcgis/rest/services/"
            "coa_Inspections_BuildingPermits_V2/MapServer/2"
        ),
        where="UPPER(DESCRIPTION) LIKE '%SOLAR%' OR UPPER(DESCRIPTION) LIKE '%PHOTOVOLT%'",
        id_field="PERMIT_NUMBER",
        description_field="DESCRIPTION",
        application_field="APPLICATION_DATE",
        issue_field="ISSUE_DATE",
        completion_field="FINALED_DATE",
        cost_field="VALUATION",
        permit_type_field="PermitType",
        work_class_field="WORK_CLASS",
        cache_name="ames_building_permits_solar.csv",
        extra_fields=("PERMIT_STATUS",),
    ),
    "bismarck_permits": PermitLayer(
        source_key="bismarck_permits",
        jurisdiction="Bismarck",
        state="ND",
        layer_url=(
            "https://services1.arcgis.com/XxHmL09eFqJWI0gE/arcgis/rest/services/"
            "Map1/FeatureServer/0"
        ),
        where="UPPER(DESCRIPTION) LIKE '%SOLAR%' OR UPPER(DESCRIPTION) LIKE '%PHOTOVOLT%'",
        id_field="PERMIT_NO",
        description_field="DESCRIPTION",
        application_field="APPLIED",
        issue_field="ISSUED",
        completion_field="FINALED",
        cost_field="JOBVALUE",
        fee_field="FEES_CHARGED",
        contractor_field="CONTRACTOR_NAME",
        permit_type_field="PermitType",
        work_class_field="PermitSubType",
        zip_field="SITE_ZIP",
        lat_field="LAT",
        lon_field="LON",
        cache_name="bismarck_building_permits_solar.csv",
    ),
}


def _col(df: pd.DataFrame, name: str | None, default=np.nan):
    if name and name in df.columns:
        return df[name]
    return pd.Series(default, index=df.index)


def fetch_raw(spec: PermitLayer, refresh: bool = False) -> pd.DataFrame:
    return arcgis.query(
        spec.layer_url,
        where=spec.where,
        out_fields=spec.out_fields,
        order_by=spec.id_field,
        date_fields=spec.date_fields,
        cache_path=RAW_DIR / spec.cache_name,
        refresh=refresh,
    )


def to_canonical(spec: PermitLayer, raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    if df.empty:
        return pd.DataFrame(columns=list(CANONICAL_COLUMNS))

    desc = _col(df, spec.description_field, "").astype("string").fillna("")
    work = _col(df, spec.work_class_field, "").astype("string").fillna("")
    blob = (desc + " " + work).str.upper()

    keep = blob.str.contains(PV_RE) & ~blob.str.contains(NON_PV_RE)
    LOG.info("%s: %d/%d rows kept as PV", spec.jurisdiction, int(keep.sum()), len(df))
    df = df.loc[keep].copy()
    df = df.drop_duplicates(subset=[spec.id_field], keep="first")
    if df.empty:
        return pd.DataFrame(columns=list(CANONICAL_COLUMNS))

    text = _col(df, spec.description_field, "").astype("string").fillna("").str.upper()
    contractor = _col(df, spec.contractor_field, np.nan).map(normalize_contractor)
    district = _col(df, spec.district_field, np.nan)
    if spec.district_field:
        district = "d-" + district.astype("string").str.replace(r"\.0$", "", regex=True)

    out = pd.DataFrame(
        {
            "record_id": spec.source_key + ":" + df[spec.id_field].astype(str),
            "source": spec.source_key,
            "jurisdiction": spec.jurisdiction,
            "state": spec.state,
            "application_date": pd.to_datetime(
                _col(df, spec.application_field, pd.NaT), errors="coerce"
            ),
            "issue_date": pd.to_datetime(
                _col(df, spec.issue_field, pd.NaT), errors="coerce"
            ),
            "completion_date": pd.to_datetime(
                _col(df, spec.completion_field, pd.NaT), errors="coerce"
            ),
            "system_size_kw": extract_kw(text).where(lambda s: (s > 0.2) & (s <= 1000)),
            "reported_cost_usd": pd.to_numeric(
                _col(df, spec.cost_field), errors="coerce"
            ),
            "permit_fee_usd": pd.to_numeric(_col(df, spec.fee_field), errors="coerce"),
            "sector": "unknown",
            "review_track": _col(df, spec.permit_type_field, "Unknown")
            .astype("string")
            .fillna("Unknown"),
            "contractor": contractor,
            "zip_code": _col(df, spec.zip_field, pd.NA).astype("string").str.slice(0, 5),
            "latitude": pd.to_numeric(_col(df, spec.lat_field), errors="coerce"),
            "longitude": pd.to_numeric(_col(df, spec.lon_field), errors="coerce"),
            "mount_type": np.where(is_ground_mount(text), "ground", "roof"),
            "roof_type": "unknown",
            "battery": has_battery(text).astype(int),
            "permit_class": work.loc[df.index].astype("string"),
            "district": district,
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


def load_layer(source_key: str, refresh: bool = False) -> pd.DataFrame:
    spec = LAYERS[source_key]
    return to_canonical(spec, fetch_raw(spec, refresh=refresh))


def make_module_api(source_key: str):
    """Return ``(SOURCE_KEY, load)`` so each layer can be registered as a source."""

    def _load(refresh: bool = False) -> pd.DataFrame:
        return load_layer(source_key, refresh=refresh)

    return source_key, _load


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for key in LAYERS:
        f = load_layer(key)
        a2i = (f.issue_date - f.application_date).dt.days.dropna()
        p2c = (f.completion_date - f.issue_date).dt.days.dropna()
        print(
            f"{key:24} rows={len(f):5}  app->issue n={len(a2i):5} med="
            f"{a2i.median() if len(a2i) else float('nan'):6.1f}"
            f"  issue->done n={len(p2c):5} med="
            f"{p2c.median() if len(p2c) else float('nan'):6.1f}"
        )
