"""Minimal client for ArcGIS REST FeatureServer layers.

Reference: https://developers.arcgis.com/rest/services-reference/enterprise/query-feature-service-layer/

Municipal open-data portals built on ArcGIS Hub expose their permit registers as
FeatureServer layers. The query endpoint is anonymous and needs no key, but it
caps each response (usually 1000-2000 features) and signals truncation with
``exceededTransferLimit``; we page with ``resultOffset``/``resultRecordCount``.

Dates come back as epoch **milliseconds** in UTC when ``f=json``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests

LOG = logging.getLogger(__name__)

PAGE_SIZE = 2000
_TIMEOUT = 180
_MAX_RETRIES = 4


def _get_json(url: str, params: dict) -> dict:
    last: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=_TIMEOUT)
            resp.raise_for_status()
            payload = resp.json()
            if "error" in payload:
                raise RuntimeError(f"ArcGIS error: {payload['error']}")
            return payload
        except Exception as exc:  # noqa: BLE001
            last = exc
            LOG.warning("ArcGIS request failed (%s); retry %d", exc, attempt + 1)
            time.sleep(2**attempt)
    raise RuntimeError(f"ArcGIS request failed after {_MAX_RETRIES} attempts") from last


def parse_esri_date(series: pd.Series) -> pd.Series:
    """Normalise an ArcGIS date column to naive daily timestamps.

    ArcGIS returns ``esriFieldTypeDate`` as epoch **milliseconds** but
    ``esriFieldTypeDateOnly`` as an ISO ``YYYY-MM-DD`` **string**, and different
    layers use different types for the same concept (Detroit's BSEED layer uses
    DateOnly; Minneapolis uses Date). Handle both rather than guessing.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        out = pd.to_datetime(numeric, unit="ms", errors="coerce")
        # rows that were not numeric may still be parseable strings
        leftover = out.isna() & series.notna()
        if leftover.any():
            out.loc[leftover] = pd.to_datetime(
                series[leftover], errors="coerce", format="mixed"
            )
        return out.dt.normalize()
    return pd.to_datetime(series, errors="coerce", format="mixed").dt.normalize()


def describe(layer_url: str) -> dict:
    """Return the layer metadata document (field names, types, row cap)."""
    return _get_json(layer_url, {"f": "json"})


def count(layer_url: str, where: str = "1=1") -> int:
    payload = _get_json(
        f"{layer_url}/query",
        {"where": where, "returnCountOnly": "true", "f": "json"},
    )
    return int(payload.get("count", 0))


def query(
    layer_url: str,
    *,
    where: str = "1=1",
    out_fields: str = "*",
    order_by: str,
    date_fields: tuple[str, ...] = (),
    cache_path: Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Page through a layer and return the attribute table as a DataFrame.

    ``order_by`` is mandatory so that offset paging is deterministic.
    """
    if cache_path is not None and cache_path.exists() and not refresh:
        LOG.info("Using cached %s", cache_path)
        df = pd.read_csv(cache_path, low_memory=False)
        for c in date_fields:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
        return df

    rows: list[dict] = []
    offset = 0
    while True:
        payload = _get_json(
            f"{layer_url}/query",
            {
                "where": where,
                "outFields": out_fields,
                "orderByFields": order_by,
                "returnGeometry": "false",
                "resultOffset": str(offset),
                "resultRecordCount": str(PAGE_SIZE),
                "f": "json",
            },
        )
        feats = payload.get("features", [])
        LOG.info("%s: %d features at offset %d", layer_url.rsplit("/", 2)[-2], len(feats), offset)
        rows.extend(f["attributes"] for f in feats)
        if len(feats) < PAGE_SIZE and not payload.get("exceededTransferLimit"):
            break
        if not feats:
            break
        offset += len(feats)

    df = pd.DataFrame(rows)
    for c in date_fields:
        if c in df.columns:
            df[c] = parse_esri_date(df[c])
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
        LOG.info("Cached %d rows -> %s", len(df), cache_path)
    return df
