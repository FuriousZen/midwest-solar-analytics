"""Minimal client for ArcGIS REST FeatureServer layers.

Reference: https://developers.arcgis.com/rest/services-reference/enterprise/query-feature-service-layer/

Municipal open-data portals built on ArcGIS Hub expose their permit registers as
FeatureServer layers. The query endpoint is anonymous and needs no key, but it
caps each response (usually 1000-2000 features) and signals truncation with
``exceededTransferLimit``; we page with ``resultOffset``/``resultRecordCount``.

Dates come back as epoch milliseconds in UTC when ``f=json``.
"""

from __future__ import annotations

import json as _json
import logging
from pathlib import Path

import pandas as pd

from msa.ingest.http import MAX_ATTEMPTS as _MAX_RETRIES  # noqa: F401  (public name)
from msa.ingest.http import SourceUnavailable, request_with_retries

LOG = logging.getLogger(__name__)

PAGE_SIZE = 2000
_TIMEOUT = 180
#: Hard stop on offset paging. The largest layer in this corpus is well under
#: 100k features; anything past this means the server is not honouring
#: ``resultOffset`` and the loop would never end.
MAX_PAGES = 500


def _get_json(url: str, params: dict) -> dict:
    """One FeatureServer request.

    ArcGIS is unusual in answering HTTP 200 with an ``error`` object for
    application-level failures (bad field name, layer deleted, token expired).
    That is a permanent condition, so it is raised immediately rather than
    retried; the transport-level retrying happens one layer down.
    """
    resp = request_with_retries(
        url, params, timeout=_TIMEOUT, source=f"arcgis:{url}"
    )
    try:
        payload = resp.json()
    except (_json.JSONDecodeError, ValueError) as exc:
        raise SourceUnavailable(
            f"arcgis:{url} returned a body that is not JSON "
            f"(schema change or an error page?): {exc}",
            source=f"arcgis:{url}",
        ) from exc
    if not isinstance(payload, dict):
        raise SourceUnavailable(
            f"arcgis:{url} returned {type(payload).__name__}, expected an object",
            source=f"arcgis:{url}",
        )
    if "error" in payload:
        raise SourceUnavailable(
            f"arcgis:{url} returned an application error: {payload['error']}",
            source=f"arcgis:{url}",
        )
    return payload


def parse_esri_date(series: pd.Series) -> pd.Series:
    """Normalise an ArcGIS date column to naive daily timestamps.

    ArcGIS returns ``esriFieldTypeDate`` as epoch milliseconds but
    ``esriFieldTypeDateOnly`` as an ISO ``YYYY-MM-DD`` string, and different
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

    A layer that answers with no features raises :class:`SourceUnavailable`
    and is never cached: every layer this project queries has matching rows,
    so an empty answer means the query or the layer is broken, and caching it
    would make the failure permanent.
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
    for _page in range(MAX_PAGES):
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
        feats = payload.get("features") or []
        LOG.info("%s: %d features at offset %d", layer_url.rsplit("/", 2)[-2], len(feats), offset)
        rows.extend(f.get("attributes", {}) for f in feats if isinstance(f, dict))
        if len(feats) < PAGE_SIZE and not payload.get("exceededTransferLimit"):
            break
        if not feats:
            break
        offset += len(feats)
    else:
        # A server that silently ignores `resultOffset` would page for ever.
        raise SourceUnavailable(
            f"arcgis:{layer_url} did not terminate within {MAX_PAGES} pages "
            f"({MAX_PAGES * PAGE_SIZE} features); it is probably ignoring "
            f"resultOffset",
            source=f"arcgis:{layer_url}",
        )

    if not rows:
        raise SourceUnavailable(
            f"arcgis:{layer_url} returned no features for this query; nothing "
            f"was cached",
            source=f"arcgis:{layer_url}",
        )
    df = pd.DataFrame(rows)
    for c in date_fields:
        if c in df.columns:
            df[c] = parse_esri_date(df[c])
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
        LOG.info("Cached %d rows -> %s", len(df), cache_path)
    return df
