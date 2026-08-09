"""Ingest LBNL "Tracking the Sun" distributed-PV records for the Midwest.

Provenance
----------
Source   : Lawrence Berkeley National Laboratory, *Tracking the Sun* -- the
           public data file of U.S. distributed PV systems compiled from state
           incentive programs, utility interconnection registries and other
           public agencies.
Mirror   : NREL/DOE Open Energy Data Initiative public S3 bucket
           ``s3://oedi-data-lake/tracking-the-sun/`` (anonymous HTTPS access),
           Hive-partitioned as ``<edition>/state=<XX>/*.snappy.parquet``.
Edition  : 2024 (latest edition mirrored on OEDI at time of writing).
Licence  : public data, free to use with attribution to LBNL.

Important
---------
The public Tracking the Sun file contains exactly ONE date column,
``installation_date``. It therefore CANNOT supply a permit-to-completion
duration and is **not** used as a source of the regression target. It is used
purely as a *utility-records* context layer: rolling market-activity and
price/technology statistics keyed on (state, ZIP, month) that are joined onto
the permit records in ``msa.features``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

from msa.config import RAW_DIR

LOG = logging.getLogger(__name__)

BUCKET_URL = "https://oedi-data-lake.s3.amazonaws.com"
PREFIX = "tracking-the-sun"
EDITION = "2024"

#: Midwest states present in the 2024 edition of the public file.
#: (IN, IA, MI, NE, ND, SD are not reported; KS/MO appear only in the 2018-2020
#: editions. Verified by listing the bucket -- `python -m scripts.verify_tts_states`)
TTS_MIDWEST_STATES: tuple[str, ...] = ("IL", "MN", "OH", "WI")

USE_COLS = [
    "installation_date",
    "pv_system_size_dc",
    "total_installed_price",
    "rebate_or_grant",
    "customer_segment",
    "new_construction",
    "tracking",
    "ground_mounted",
    "zip_code",
    "city",
    "utility_service_territory",
    "third_party_owned",
    "installer_name",
    "self_installed",
    "battery_rated_capacity_kw",
    "technology_type",
    "efficiency_module_1",
    "inverter_loading_ratio",
]


def _list_state_keys(edition: str) -> dict[str, str]:
    """Map two-letter state -> S3 object key for one TTS edition."""
    import xml.etree.ElementTree as ET

    url = f"{BUCKET_URL}/?list-type=2&prefix={PREFIX}/{edition}/&max-keys=1000"
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    root = ET.fromstring(resp.text)
    out: dict[str, str] = {}
    for c in root.findall("s3:Contents", ns):
        key = c.findtext("s3:Key", default="", namespaces=ns)
        if "state=" in key and key.endswith(".parquet"):
            state = key.split("state=")[1].split("/")[0]
            out[state] = key
    return out


def download(
    states: tuple[str, ...] = TTS_MIDWEST_STATES,
    edition: str = EDITION,
    refresh: bool = False,
) -> dict[str, Path]:
    keys = _list_state_keys(edition)
    missing = [s for s in states if s not in keys]
    if missing:
        raise RuntimeError(
            f"TTS edition {edition} does not publish these states: {missing}. "
            f"Available: {sorted(keys)}"
        )
    paths: dict[str, Path] = {}
    for state in states:
        dest = RAW_DIR / f"tts_{edition}_{state}.parquet"
        if dest.exists() and not refresh:
            LOG.info("Using cached %s", dest)
        else:
            url = f"{BUCKET_URL}/{keys[state]}"
            LOG.info("Downloading %s -> %s", url, dest)
            with requests.get(url, stream=True, timeout=600) as r:
                r.raise_for_status()
                with open(dest, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
        paths[state] = dest
    return paths


def load(
    states: tuple[str, ...] = TTS_MIDWEST_STATES,
    edition: str = EDITION,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return one row per Midwest PV system with a usable installation date."""
    frames = []
    for state, path in download(states, edition, refresh).items():
        df = pd.read_parquet(path, columns=USE_COLS)
        df["state"] = state
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["installation_date"] = pd.to_datetime(out["installation_date"], errors="coerce")
    out = out.dropna(subset=["installation_date"])
    out["zip_code"] = out["zip_code"].astype("string").str.slice(0, 5)
    LOG.info(
        "Tracking the Sun (%s): %d Midwest systems, %s -> %s",
        edition,
        len(out),
        out["installation_date"].min().date(),
        out["installation_date"].max().date(),
    )
    return out


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    frame = load()
    print(frame.shape)
    print(frame.groupby("state")["installation_date"].agg(["count", "min", "max"]))
