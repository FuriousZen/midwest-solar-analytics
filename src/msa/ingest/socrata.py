"""Thin, dependency-light client for the Socrata Open Data API (SODA 2.1).

Reference: https://dev.socrata.com/docs/endpoints.html

Only the handful of features we need are implemented:

* ``$select`` / ``$where`` / ``$order`` / ``$limit`` / ``$offset`` SoQL clauses
* keyset-free offset pagination (fine at our volumes, tens of thousands of rows)
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
from pathlib import Path

import pandas as pd
import requests

from msa.ingest.http import MAX_ATTEMPTS as _MAX_RETRIES  # noqa: F401  (public name)
from msa.ingest.http import SourceUnavailable, request_with_retries

LOG = logging.getLogger(__name__)

PAGE_SIZE = 20_000
_TIMEOUT = 120
#: Hard stop on offset paging. Every dataset in this corpus is well under this
#: many pages; going past it means the portal is not honouring ``$offset``.
MAX_PAGES = 200


def _headers() -> dict[str, str]:
    token = os.environ.get("SOCRATA_APP_TOKEN")
    return {"X-App-Token": token} if token else {}


def _get(url: str, params: dict[str, str]) -> requests.Response:
    """One SODA request, with the shared timeout/backoff policy applied."""
    return request_with_retries(
        url,
        params,
        timeout=_TIMEOUT,
        headers=_headers(),
        source=f"socrata:{url}",
    )


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

    A response with no rows raises :class:`SourceUnavailable` and is never
    cached. Every dataset this project queries has matching rows, so an empty
    answer means the query or the portal is broken (a double-encoded ``%``
    in the predicate, for instance, matches nothing and returns 200), and
    caching it would make the failure permanent.
    """
    if cache_path is not None and cache_path.exists() and not refresh:
        LOG.info("Using cached %s", cache_path)
        return pd.read_csv(cache_path, low_memory=False)

    url = f"https://{domain}/resource/{dataset_id}.csv"
    frames: list[pd.DataFrame] = []
    offset = 0
    for _page in range(MAX_PAGES):
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
        # An empty body is a legitimate answer (we paged past the end) and is
        # not the same thing as a broken one. pandas raises on both, so
        # distinguish them here rather than letting EmptyDataError escape as if
        # the source were down.
        stripped = text.lstrip()
        if not stripped:
            LOG.info("%s/%s: empty page at offset %d", domain, dataset_id, offset)
            break
        # A portal in trouble answers 200 with an HTML error page. pandas will
        # happily parse that into a one-column frame of junk, which then flows
        # all the way into the corpus. Detect it here, where it is obvious.
        if stripped[0] == "<":
            raise SourceUnavailable(
                f"socrata:{domain}/{dataset_id} returned HTML, not CSV "
                f"(error page behind a 200?)",
                source=f"socrata:{domain}/{dataset_id}",
            )
        try:
            page = pd.read_csv(io.StringIO(text), low_memory=False)
        except pd.errors.EmptyDataError:
            LOG.info(
                "%s/%s: header-only page at offset %d", domain, dataset_id, offset
            )
            break
        except pd.errors.ParserError as exc:
            raise SourceUnavailable(
                f"socrata:{domain}/{dataset_id} returned a body that is not CSV "
                f"(schema change or an error page?): {exc}",
                source=f"socrata:{domain}/{dataset_id}",
            ) from exc
        LOG.info("%s/%s: fetched %d rows at offset %d", domain, dataset_id, len(page), offset)
        if page.empty:
            break
        frames.append(page)
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    else:
        raise SourceUnavailable(
            f"socrata:{domain}/{dataset_id} did not terminate within {MAX_PAGES} "
            f"pages ({MAX_PAGES * PAGE_SIZE} rows); the portal is probably "
            f"ignoring $offset",
            source=f"socrata:{domain}/{dataset_id}",
        )

    if not frames:
        raise SourceUnavailable(
            f"socrata:{domain}/{dataset_id} returned no rows for this query; "
            f"nothing was cached",
            source=f"socrata:{domain}/{dataset_id}",
        )
    df = pd.concat(frames, ignore_index=True)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
        LOG.info("Cached %d rows -> %s", len(df), cache_path)
    return df
