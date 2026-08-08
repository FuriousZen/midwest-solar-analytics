"""Thin, dependency-light client for the Socrata Open Data API (SODA 2.1).

Reference: https://dev.socrata.com/docs/endpoints.html

Only the handful of features we need are implemented:

* ``$select`` / ``$where`` / ``$order`` / ``$limit`` / ``$offset`` SoQL clauses
* keyset-free offset pagination (fine at our volumes -- tens of thousands of rows)
* on-disk caching of the raw CSV so the rest of the pipeline is reproducible
  offline once the download has happened at least once.

No API token is required for the request volumes used here; Socrata throttles
anonymous clients but does not block them. Set ``SOCRATA_APP_TOKEN`` in the
environment to raise the limit.
"""

from __future__ import annotations

import io
import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests

LOG = logging.getLogger(__name__)

PAGE_SIZE = 20_000
_TIMEOUT = 120
_MAX_RETRIES = 4


def _headers() -> dict[str, str]:
    token = os.environ.get("SOCRATA_APP_TOKEN")
    return {"X-App-Token": token} if token else {}


def _get(url: str, params: dict[str, str]) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(
                url, params=params, headers=_headers(), timeout=_TIMEOUT
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                raise requests.HTTPError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            resp.raise_for_status()
            return resp
        except Exception as exc:  # noqa: BLE001 - retry any transport error
            last_exc = exc
            sleep = 2**attempt
            LOG.warning("Socrata request failed (%s); retrying in %ss", exc, sleep)
            time.sleep(sleep)
    raise RuntimeError(f"Socrata request failed after {_MAX_RETRIES} attempts") from last_exc


def count(domain: str, dataset_id: str, where: str | None = None) -> int:
    """Return the number of rows matching ``where``."""
    url = f"https://{domain}/resource/{dataset_id}.json"
    params = {"$select": "count(1) as n"}
    if where:
        params["$where"] = where
    payload = _get(url, params).json()
    return int(payload[0]["n"])


def fetch_csv(
    domain: str,
    dataset_id: str,
    *,
    select: str | None = None,
    where: str | None = None,
    order: str,
    cache_path: Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Download every row matching ``where`` as a DataFrame.

    ``order`` is mandatory: Socrata does not guarantee a stable row order across
    paginated requests unless an explicit ``$order`` is supplied, which would
    silently produce duplicated/missing rows.
    """
    if cache_path is not None and cache_path.exists() and not refresh:
        LOG.info("Using cached %s", cache_path)
        return pd.read_csv(cache_path, low_memory=False)

    url = f"https://{domain}/resource/{dataset_id}.csv"
    frames: list[pd.DataFrame] = []
    offset = 0
    while True:
        params: dict[str, str] = {
            "$order": order,
            "$limit": str(PAGE_SIZE),
            "$offset": str(offset),
        }
        if select:
            params["$select"] = select
        if where:
            params["$where"] = where
        text = _get(url, params).text
        page = pd.read_csv(io.StringIO(text), low_memory=False)
        LOG.info("%s/%s: fetched %d rows at offset %d", domain, dataset_id, len(page), offset)
        if page.empty:
            break
        frames.append(page)
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
        LOG.info("Cached %d rows -> %s", len(df), cache_path)
    return df
