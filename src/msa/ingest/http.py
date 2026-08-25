"""Shared retry/timeout policy for the two public-API clients.

Both the Socrata and the ArcGIS client talk to municipal portals that are
occasionally slow, occasionally 500, and occasionally rate-limit. The policy is
the same for both, so it lives in one place:

* Every request has a timeout. A hung socket is the one failure a retry loop
  cannot recover from, because it never returns to be retried.
* Retry only what retrying can fix: connection errors, timeouts, 429, and 5xx.
  A 404 (dataset moved), a 400 (bad SoQL) or a 403 (token revoked) will return
  exactly the same answer four times over; failing immediately turns a
  ~15-second wait into an instant, accurate error message.
* Exponential backoff with full jitter, and no sleep after the final attempt.
  An earlier version slept 8 seconds before raising, which is pure latency
  added to a failure that had already happened.
* One exception type (:class:`SourceUnavailable`) so callers can isolate a dead
  jurisdiction without catching bare ``Exception``.
"""

from __future__ import annotations

import logging
import random
import time

import requests

LOG = logging.getLogger(__name__)

#: Attempts, not retries: 4 means one try plus three retries.
MAX_ATTEMPTS = 4
#: Base for the exponential backoff, in seconds.
BACKOFF_BASE = 1.0
#: Never wait longer than this between attempts.
BACKOFF_CAP = 30.0

#: Status codes where trying again is a reasonable thing to do.
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504, 509})


class SourceUnavailable(RuntimeError):
    """A public data source could not be read after the retry budget ran out.

    Carries ``source`` so ``build_dataset`` can name the jurisdiction it had to
    skip without parsing a message string.
    """

    def __init__(self, message: str, *, source: str = "", status: int | None = None):
        super().__init__(message)
        self.source = source
        self.status = status


class NonRetryableStatus(SourceUnavailable):
    """The server answered, definitively, that this request will never work."""


def _sleep_for(attempt: int) -> float:
    """Full-jitter exponential backoff (AWS Architecture Blog's formulation).

    Jitter matters here because ``build_dataset`` hits several layers on the
    same ArcGIS host; without it, a shared outage produces synchronised retries.
    """
    ceiling = min(BACKOFF_CAP, BACKOFF_BASE * (2**attempt))
    return random.uniform(0.0, ceiling)


def request_with_retries(
    url: str,
    params: dict | None = None,
    *,
    timeout: float,
    headers: dict[str, str] | None = None,
    source: str = "",
    max_attempts: int = MAX_ATTEMPTS,
    sleep=None,
) -> requests.Response:
    """GET ``url``, retrying only transient failures. Never returns a bad status.

    ``sleep`` is resolved at call time, not bound as a default, so a test can
    replace ``time.sleep`` and actually observe the backoff schedule.
    """
    sleep = sleep or time.sleep
    last: Exception | None = None
    for attempt in range(max_attempts):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
        except requests.RequestException as exc:
            # connection reset, DNS failure, read timeout: all transient
            last = exc
            LOG.warning(
                "%s: transport error on attempt %d/%d (%s)",
                source or url,
                attempt + 1,
                max_attempts,
                exc,
            )
        else:
            if resp.status_code < 400:
                return resp
            if resp.status_code not in RETRYABLE_STATUS:
                raise NonRetryableStatus(
                    f"{source or url}: HTTP {resp.status_code} is not retryable "
                    f"({resp.reason}); the request itself is wrong or the "
                    f"dataset has moved",
                    source=source,
                    status=resp.status_code,
                )
            last = requests.HTTPError(f"HTTP {resp.status_code} {resp.reason}")
            LOG.warning(
                "%s: HTTP %d on attempt %d/%d",
                source or url,
                resp.status_code,
                attempt + 1,
                max_attempts,
            )
        # no backoff after the final attempt; it would delay a failure that
        # has already been decided
        if attempt < max_attempts - 1:
            sleep(_sleep_for(attempt))

    raise SourceUnavailable(
        f"{source or url}: failed after {max_attempts} attempts ({last})",
        source=source,
    ) from last


__all__ = [
    "BACKOFF_CAP",
    "MAX_ATTEMPTS",
    "RETRYABLE_STATUS",
    "NonRetryableStatus",
    "SourceUnavailable",
    "request_with_retries",
]
