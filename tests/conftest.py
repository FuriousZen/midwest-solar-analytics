"""Shared fixtures.

Tests never hit the network. Anything that needs canonical records builds them
from ``synthetic_canonical`` -- a hand-written frame with known properties, used
only to exercise transformation logic. It is never used for training or for any
number that appears in the README.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from msa.config import CANONICAL_COLUMNS


@pytest.fixture
def synthetic_canonical() -> pd.DataFrame:
    """Deterministic canonical records for unit-testing transformations."""
    rng = np.random.default_rng(0)
    n = 400
    start = pd.Timestamp("2021-01-01") + pd.to_timedelta(
        rng.integers(0, 900, n), unit="D"
    )
    review = rng.integers(1, 40, n)
    build = rng.integers(5, 200, n)
    contractors = np.array(["ACME SOLAR", "BRIGHT ROOF", "CEDAR PV", None], dtype=object)

    df = pd.DataFrame(
        {
            "record_id": [f"test:{i}" for i in range(n)],
            "source": "test_source",
            "jurisdiction": rng.choice(["Testville", "Sampleburg"], n),
            "state": "IL",
            "application_date": start,
            "issue_date": start + pd.to_timedelta(review, unit="D"),
            "completion_date": start + pd.to_timedelta(review + build, unit="D"),
            "system_size_kw": rng.uniform(3, 25, n),
            "reported_cost_usd": rng.uniform(8000, 80000, n),
            "permit_fee_usd": rng.uniform(50, 900, n),
            "sector": rng.choice(["residential", "commercial"], n),
            "review_track": rng.choice(["Express", "Standard"], n),
            "contractor": rng.choice(contractors, n),
            "zip_code": rng.choice(["60601", "60602", "55401"], n),
            "latitude": rng.uniform(41, 45, n),
            "longitude": rng.uniform(-93, -87, n),
            "mount_type": "roof",
            "roof_type": rng.choice(["flat", "pitched"], n),
            "battery": rng.integers(0, 2, n),
            "permit_class": rng.choice(["Alteration", "New"], n),
            "district": rng.choice(["ward-1", "ward-2"], n),
            "building_fee_usd": rng.uniform(0, 500, n),
            "zoning_fee_usd": rng.choice([0.0, 50.0], n),
            "other_fee_usd": rng.uniform(0, 300, n),
            "n_parties": rng.integers(1, 6, n),
            "has_expeditor": rng.integers(0, 2, n),
            "has_engineer": rng.integers(0, 2, n),
            "has_design_professional": rng.integers(0, 2, n),
            "description_length": rng.integers(20, 400, n),
        }
    )
    return df[list(CANONICAL_COLUMNS)]
