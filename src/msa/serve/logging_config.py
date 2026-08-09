"""Structured JSON logging with request correlation.

Every log line the service emits is a single JSON object on one line, so a log
shipper can index it without a regex. The current request's id is carried in a
:class:`contextvars.ContextVar` and stamped onto every record emitted while
handling that request -- including records from libraries that know nothing
about the web layer (xgboost, pandas, our own feature code).

``contextvars`` is the right primitive here rather than a thread-local: FastAPI
runs ``def`` endpoints in a worker thread but ``async def`` endpoints on the
event loop, and ``contextvars`` is the only mechanism that is correct for both.
Starlette copies the context per request, and ``anyio.to_thread.run_sync``
propagates it into the worker thread.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import sys
import uuid
from contextvars import ContextVar

#: Correlation id for the request currently being handled, if any.
request_id_var: ContextVar[str | None] = ContextVar("msa_request_id", default=None)

#: Fields ``logging.LogRecord`` always defines; anything else on a record was
#: put there deliberately by ``logger.x(..., extra={...})`` and is worth keeping.
_STANDARD_RECORD_FIELDS = frozenset(
    vars(logging.LogRecord("", 0, "", 0, "", (), None)).keys()
) | {"message", "asctime", "taskName"}


def new_request_id() -> str:
    return uuid.uuid4().hex


class JsonFormatter(logging.Formatter):
    """One JSON object per line, with the correlation id folded in."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": _dt.datetime.fromtimestamp(
                record.created, tz=_dt.timezone.utc
            ).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        rid = getattr(record, "request_id", None) or request_id_var.get()
        if rid:
            payload["request_id"] = rid
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key in _STANDARD_RECORD_FIELDS or key == "request_id":
                continue
            try:
                json.dumps(value)
            except (TypeError, ValueError):
                value = repr(value)
            payload[key] = value
        return json.dumps(payload, default=str)


def configure_logging(level: str | int | None = None) -> None:
    """Install the JSON formatter on the root logger.

    Idempotent: calling it twice does not double every line. Set
    ``MSA_LOG_FORMAT=text`` to fall back to human-readable logs when developing.
    """
    level = level or os.environ.get("MSA_LOG_LEVEL", "INFO")
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, "_msa_json", False):
            handler.setLevel(level)
            root.setLevel(level)
            return

    handler = logging.StreamHandler(sys.stdout)
    if os.environ.get("MSA_LOG_FORMAT", "json").lower() == "text":
        handler.setFormatter(
            logging.Formatter("%(levelname)s %(name)s: %(message)s")
        )
    else:
        handler.setFormatter(JsonFormatter())
    handler._msa_json = True  # type: ignore[attr-defined]
    root.handlers = [h for h in root.handlers if not isinstance(h, logging.StreamHandler)]
    root.addHandler(handler)
    root.setLevel(level)
    # uvicorn installs its own handlers; let ours be the only sink so the log
    # stream stays uniformly JSON.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True


__all__ = ["JsonFormatter", "configure_logging", "new_request_id", "request_id_var"]
