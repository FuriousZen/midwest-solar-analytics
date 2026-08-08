"""Tests for the ingestion layer.

These exercise the parsing and normalisation logic on fixed inputs -- no
network calls, so the suite runs offline and deterministically. Each ingester's
``to_canonical`` is fed a miniature version of its real source payload.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from msa.config import CANONICAL_COLUMNS
from msa.ingest import arcgis, chicago
from msa.ingest.arcgis_permits import LAYERS, PermitLayer, to_canonical
from msa.ingest.text import extract_kw, has_battery, is_pv, normalize_contractor


# --------------------------------------------------------------------------
# shared text helpers
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("INSTALL SOLAR PANELS ON ROOF", True),
        ("ERECT A PHOTOVOLTAIC ARRAY", True),
        ("ROOFTOP SOLAR PV SYSTEM 7.4 KW", True),
        ("CONSTRUCT NEW SOLARIUM OFF KITCHEN", False),
        ("INSTALL SOLAR HOT WATER PANELS", False),
        ("REPLACE SOLATUBE SKYLIGHT", False),
        ("APPLY SOLAR CONTROL WINDOW FILM", False),
        ("REROOF GARAGE", False),
    ],
)
def test_pv_classification(text, expected):
    assert bool(is_pv(pd.Series([text])).iloc[0]) is expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("All Energy Solar, Inc.", "ALL ENERGY SOLAR"),
        ("ALL ENERGY SOLAR LLC", "ALL ENERGY SOLAR"),
        ("Wolf River Electric Co.", "WOLF RIVER ELECTRIC"),
        ("", None),
        (None, None),
        (12345, None),
    ],
)
def test_contractor_normalisation_collapses_variants(raw, expected):
    got = normalize_contractor(raw)
    if expected is None:
        assert got is np.nan or pd.isna(got)
    else:
        assert got == expected


def test_kw_extraction():
    s = pd.Series(["ARRAY OUTPUT: 14.55 KW", "7KW SYSTEM", "NO SIZE HERE"])
    got = extract_kw(s.str.upper())
    assert got.iloc[0] == pytest.approx(14.55)
    assert got.iloc[1] == pytest.approx(7.0)
    assert pd.isna(got.iloc[2])


def test_battery_detection():
    s = pd.Series(["PV WITH BATTERY BACKUP", "PV + POWERWALL", "PV ONLY"]).str.upper()
    assert has_battery(s).tolist() == [True, True, False]


# --------------------------------------------------------------------------
# Chicago
# --------------------------------------------------------------------------


def _chicago_row(**over) -> dict:
    row = {
        "id": "N1",
        "permit_": "B1",
        "permit_status": "COMPLETE",
        "permit_milestone": "COMPLETE",
        "permit_type": "PERMIT - EASY PERMIT PROCESS",
        "review_type": "EASY PERMIT",
        "application_start_date": "2024-01-02T00:00:00.000",
        "issue_date": "2024-01-09T00:00:00.000",
        "processing_time": 7,
        "work_type": "Small-Scale Solar PV System",
        "work_description": (
            "INSTALL SOLAR PANELS ON ROOF. ROOF TYPE: SLOPED (2:12 TO 5:12). "
            "INVERTER OUTPUT: 7.6 KW. ARRAY OUTPUT: 9.12 KW. WITH BATTERY."
        ),
        "reported_cost": 30000,
        "total_fee": 275,
        "building_fee_subtotal": 225,
        "zoning_fee_subtotal": 50,
        "other_fee_subtotal": 0,
        "community_area": 22,
        "census_tract": 220600,
        "ward": 35.0,
        "latitude": 41.9,
        "longitude": -87.7,
    }
    for i in range(1, 16):
        row[f"contact_{i}_type"] = None
        row[f"contact_{i}_name"] = None
        row[f"contact_{i}_zipcode"] = None
    row["contact_1_type"] = "OWNER"
    row["contact_1_name"] = "A HOMEOWNER"
    row["contact_1_zipcode"] = "606471234"
    row["contact_2_type"] = "GENERAL CONTRACTOR"
    row["contact_2_name"] = "Windfree Solar, LLC"
    row["contact_3_type"] = "EXPEDITOR"
    row["contact_3_name"] = "FAST PERMITS"
    row.update(over)
    return row


def test_chicago_canonicalisation_parses_description():
    raw = pd.DataFrame([_chicago_row()])
    out = chicago.to_canonical(raw)
    assert list(out.columns) == list(CANONICAL_COLUMNS)
    r = out.iloc[0]
    assert r["state"] == "IL" and r["jurisdiction"] == "Chicago"
    assert r["application_date"] == pd.Timestamp("2024-01-02")
    assert r["issue_date"] == pd.Timestamp("2024-01-09")
    assert pd.isna(r["completion_date"])  # Chicago has no completion date
    assert r["system_size_kw"] == pytest.approx(9.12)  # array, not inverter
    assert r["roof_type"] == "pitched"
    assert r["battery"] == 1
    assert r["contractor"] == "WINDFREE SOLAR"
    assert r["zip_code"] == "60647"
    assert r["has_expeditor"] == 1
    assert r["district"] == "ward-35"
    assert r["zoning_fee_usd"] == 50


def test_chicago_rejects_non_pv_rows():
    raw = pd.DataFrame(
        [
            _chicago_row(
                id="N2", work_type=None, work_description="BUILD A SOLARIUM ADDITION"
            ),
            _chicago_row(
                id="N3",
                work_type=None,
                work_description="INSTALL SOLAR HOT WATER COLLECTORS",
            ),
            _chicago_row(id="N4"),
        ]
    )
    out = chicago.to_canonical(raw)
    assert out["record_id"].tolist() == ["chicago_permits:N4"]


def test_chicago_prefers_array_over_inverter_output():
    raw = pd.DataFrame(
        [
            _chicago_row(
                work_description="SOLAR PANELS. INVERTER OUTPUT: 3.8 KW. ARRAY OUTPUT: 5.2 KW."
            )
        ]
    )
    assert chicago.to_canonical(raw).iloc[0]["system_size_kw"] == pytest.approx(5.2)


# --------------------------------------------------------------------------
# ArcGIS
# --------------------------------------------------------------------------


def test_parse_esri_date_handles_epoch_ms_and_iso_strings():
    ms = arcgis.parse_esri_date(pd.Series([1_600_000_000_000, None]))
    assert ms.iloc[0] == pd.Timestamp("2020-09-13")
    assert pd.isna(ms.iloc[1])

    iso = arcgis.parse_esri_date(pd.Series(["2020-10-28", "2022-02-23", None]))
    assert iso.iloc[0] == pd.Timestamp("2020-10-28")
    assert iso.iloc[1] == pd.Timestamp("2022-02-23")
    assert pd.isna(iso.iloc[2])


def test_arcgis_spec_canonicalisation():
    spec = LAYERS["naperville_permits"]
    raw = pd.DataFrame(
        [
            {
                "PERMITNUMBER": "P1",
                "DESCRIPTION": "INSTALL 8.4 KW ROOFTOP SOLAR PV SYSTEM",
                "APPLYDATE": pd.Timestamp("2024-03-01"),
                "ISSUEDATE": pd.Timestamp("2024-03-20"),
                "FINALIZEDATE": pd.Timestamp("2024-06-10"),
                "PERMITVALUATION": 24000,
                "PERMITTYPE": "Residential",
                "PERMITWORKCLASS": "SOLAR/PHOTOVOLTAIC",
                "POSTALCODE": "60540",
                "PERMITSTATUS": "Finalized",
            },
            {
                "PERMITNUMBER": "P2",
                "DESCRIPTION": "ENCLOSE SOLARIUM",
                "APPLYDATE": pd.Timestamp("2024-03-01"),
                "ISSUEDATE": pd.Timestamp("2024-03-20"),
                "FINALIZEDATE": None,
                "PERMITVALUATION": 1000,
                "PERMITTYPE": "Residential",
                "PERMITWORKCLASS": "ADDITION",
                "POSTALCODE": "60540",
                "PERMITSTATUS": "Issued",
            },
        ]
    )
    out = to_canonical(spec, raw)
    assert out["record_id"].tolist() == ["naperville_permits:P1"]
    r = out.iloc[0]
    assert r["system_size_kw"] == pytest.approx(8.4)
    assert r["state"] == "IL"
    assert r["completion_date"] == pd.Timestamp("2024-06-10")


def test_every_arcgis_spec_declares_at_least_two_milestones():
    for key, spec in LAYERS.items():
        assert isinstance(spec, PermitLayer)
        assert len(spec.date_fields) >= 2, key
        assert spec.id_field in spec.out_fields
        assert spec.description_field in spec.out_fields
