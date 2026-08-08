"""Central configuration: paths, constants, and source registry.

All paths are anchored to the repository root so that scripts can be run from
anywhere.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("MSA_DATA_DIR", REPO_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = Path(os.environ.get("MSA_MODELS_DIR", REPO_ROOT / "models"))
REPORTS_DIR = Path(os.environ.get("MSA_REPORTS_DIR", REPO_ROOT / "reports"))

# MLflow tracking store. Kept inside the repo (gitignored); a summary export is
# committed instead -- see scripts/export_mlflow_summary.py
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", f"file://{REPO_ROOT / 'mlruns'}"
)
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "midwest-solar-timeline")

for _d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------

#: US Census Bureau "Midwest" region (Divisions 3 and 4).
MIDWEST_STATES: tuple[str, ...] = (
    "IL",
    "IN",
    "IA",
    "KS",
    "MI",
    "MN",
    "MO",
    "NE",
    "ND",
    "OH",
    "SD",
    "WI",
)

#: Canonical column layout produced by every ingester in ``msa.ingest``.
#: ``build_dataset`` concatenates sources on exactly these columns.
CANONICAL_COLUMNS: tuple[str, ...] = (
    "record_id",  # unique id within source
    "source",  # short source key, e.g. "chicago_permits"
    "jurisdiction",  # city / county / agency name
    "state",  # 2-letter state code
    "start_date",  # timeline start (permit application / interconnection request)
    "end_date",  # timeline end (permit issuance / completion / PTO)
    "timeline_days",  # (end_date - start_date).days  <- REGRESSION TARGET
    "system_size_kw",  # DC array size where reported
    "reported_cost_usd",  # reported job cost / valuation where reported
    "permit_fee_usd",
    "sector",  # residential / commercial / unknown
    "review_track",  # source-specific review pathway (express / standard / ...)
    "contractor",  # normalised installer / general contractor name
    "zip_code",
    "latitude",
    "longitude",
    "mount_type",  # roof / ground / unknown
    "roof_type",  # flat / pitched / unknown
    "battery",  # 1 / 0 / NaN
)

#: The regression target.
TARGET = "timeline_days"

#: Records outside this window are dropped as data-entry errors / non-comparable.
#: Documented in README; 0 days is legitimate (same-day over-the-counter permits).
MIN_TIMELINE_DAYS = 0
MAX_TIMELINE_DAYS = 365

RANDOM_SEED = 20250201
