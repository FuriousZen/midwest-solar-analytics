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
    # --- the three milestone dates of a distributed-solar project ------------
    # A source supplies whichever of these it publishes; missing ones are NaT.
    "application_date",  # permit application / interconnection request filed
    "issue_date",  # permit issued / approved to build
    "completion_date",  # final inspection passed / permit closed / PTO
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
    # --- optional enrichment; sources that cannot supply these leave them NaN --
    "permit_class",  # source-specific permit type/class string
    "district",  # ward / council district / neighbourhood code
    "building_fee_usd",
    "zoning_fee_usd",
    "other_fee_usd",
    "n_parties",  # number of named parties on the application
    "has_expeditor",  # a permit expediter was retained
    "has_engineer",  # structural engineer of record named
    "has_design_professional",  # architect / registered design professional named
    "description_length",  # characters of scope-of-work text
)

#: The regression target.
TARGET = "timeline_days"

#: Which pair of milestone dates ``timeline_days`` measures.
#:
#: ``permit_to_completion``  issue_date -> completion_date. The field
#:     installation timeline: how long from "you may build" to "it is built and
#:     signed off". This is the segment the résumé line means by "installation
#:     timeline" and it is the project's primary target.
#: ``application_to_issue`` application_date -> issue_date. Permit review time.
#: ``application_to_completion`` the whole chain, only available where a source
#:     publishes all three dates.
#: ``pooled`` every available segment stacked into one frame with a
#:     ``timeline_segment`` feature identifying which milestones each row spans.
SEGMENTS: dict[str, tuple[str, str]] = {
    "permit_to_completion": ("issue_date", "completion_date"),
    "application_to_issue": ("application_date", "issue_date"),
    "application_to_completion": ("application_date", "completion_date"),
}

DEFAULT_SEGMENT = os.environ.get("MSA_SEGMENT", "pooled")

#: Modelling horizon per segment. 0 days is legitimate (same-day
#: over-the-counter permits, or an inspection closed the day it was issued).
#: The upper bound is the horizon the service predicts within; jobs that take
#: longer are treated as out of scope rather than silently squashed into the
#: target distribution. Each bound covers >=98% of the observed rows for that
#: segment -- see reports/dataset_summary.json for the realised percentages.
MIN_TIMELINE_DAYS = 0
MAX_TIMELINE_DAYS_BY_SEGMENT: dict[str, int] = {
    "application_to_issue": 120,
    "permit_to_completion": 730,
    "application_to_completion": 730,
}

#: Right-censoring guard. A project whose clock started less than
#: ``MAX_TIMELINE_DAYS`` before the data snapshot can only appear in an
#: open-data extract if it *already* finished, so the tail of any download is
#: biased towards short timelines. We drop rows whose start date falls after
#: ``max(end_date) - horizon`` for that source/segment.
def max_timeline_days(segment: str) -> int:
    return MAX_TIMELINE_DAYS_BY_SEGMENT[segment]

RANDOM_SEED = 20250201
