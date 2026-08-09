"""Failure modes of the ingestion layer, exercised against fake HTTP responses.

The ingesters hit twelve live municipal portals. The questions that matter are
not "does the happy path parse" (``test_ingest.py`` covers that) but "what
happens when a portal 500s, hangs, changes its schema, or returns nothing" --
and, above all, "does one dead jurisdiction take the whole run down with it".

No test here touches the network: ``requests.get`` is replaced with a stub that
returns exactly the pathological response under test.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
import requests

from msa.ingest import arcgis, build_dataset, socrata
from msa.ingest.http import (
    MAX_ATTEMPTS,
    NonRetryableStatus,
    SourceUnavailable,
    request_with_retries,
)


class FakeResponse:
    def __init__(self, status_code=200, text="", payload=None, reason="Fake"):
        self.status_code = status_code
        self.text = text
        self._payload = payload
        self.reason = reason

    def json(self):
        if self._payload is None:
            return json.loads(self.text)
        return self._payload


class Recorder:
    """Stands in for ``requests.get`` and records every call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def __call__(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        item = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def no_sleep(monkeypatch):
    """Backoff is real but must not make the suite take 15 seconds."""
    slept: list[float] = []
    monkeypatch.setattr(
        "msa.ingest.http.time.sleep", lambda s: slept.append(s)
    )
    return slept


# ---------------------------------------------------------------------------
# retry policy
# ---------------------------------------------------------------------------


def test_every_request_carries_a_timeout(monkeypatch):
    """A hung socket is the one failure a retry loop cannot recover from."""
    rec = Recorder([FakeResponse(200, "ok")])
    monkeypatch.setattr(requests, "get", rec)
    request_with_retries("https://example.test", {}, timeout=12.5)
    assert rec.calls[0]["timeout"] == 12.5


def test_server_error_is_retried_then_raises(monkeypatch, no_sleep):
    rec = Recorder([FakeResponse(500, "boom")])
    monkeypatch.setattr(requests, "get", rec)
    with pytest.raises(SourceUnavailable):
        request_with_retries("https://example.test", {}, timeout=5, source="s")
    assert len(rec.calls) == MAX_ATTEMPTS


def test_timeout_is_retried_then_raises(monkeypatch, no_sleep):
    rec = Recorder([requests.Timeout("read timed out")])
    monkeypatch.setattr(requests, "get", rec)
    with pytest.raises(SourceUnavailable):
        request_with_retries("https://example.test", {}, timeout=5)
    assert len(rec.calls) == MAX_ATTEMPTS


def test_connection_error_is_retried(monkeypatch, no_sleep):
    rec = Recorder([requests.ConnectionError("dns failure")])
    monkeypatch.setattr(requests, "get", rec)
    with pytest.raises(SourceUnavailable):
        request_with_retries("https://example.test", {}, timeout=5)
    assert len(rec.calls) == MAX_ATTEMPTS


def test_transient_failure_then_success_returns_the_success(monkeypatch, no_sleep):
    rec = Recorder([FakeResponse(503), FakeResponse(200, "rows")])
    monkeypatch.setattr(requests, "get", rec)
    resp = request_with_retries("https://example.test", {}, timeout=5)
    assert resp.text == "rows"
    assert len(rec.calls) == 2


def test_rate_limit_is_retried(monkeypatch, no_sleep):
    rec = Recorder([FakeResponse(429), FakeResponse(200, "rows")])
    monkeypatch.setattr(requests, "get", rec)
    assert request_with_retries("https://example.test", {}, timeout=5).text == "rows"


@pytest.mark.parametrize("status", [400, 401, 403, 404, 410, 422])
def test_client_errors_are_not_retried(monkeypatch, no_sleep, status):
    """Retrying a 404 four times gives four 404s and 15 seconds of latency."""
    rec = Recorder([FakeResponse(status)])
    monkeypatch.setattr(requests, "get", rec)
    with pytest.raises(NonRetryableStatus) as excinfo:
        request_with_retries("https://example.test", {}, timeout=5, source="s")
    assert len(rec.calls) == 1
    assert excinfo.value.status == status
    assert no_sleep == []


def test_no_backoff_is_slept_after_the_final_attempt(monkeypatch, no_sleep):
    rec = Recorder([FakeResponse(500)])
    monkeypatch.setattr(requests, "get", rec)
    with pytest.raises(SourceUnavailable):
        request_with_retries("https://example.test", {}, timeout=5)
    assert len(no_sleep) == MAX_ATTEMPTS - 1


def test_backoff_grows_and_is_jittered(monkeypatch, no_sleep):
    rec = Recorder([FakeResponse(500)])
    monkeypatch.setattr(requests, "get", rec)
    with pytest.raises(SourceUnavailable):
        request_with_retries("https://example.test", {}, timeout=5, max_attempts=5)
    assert all(0.0 <= s <= 30.0 for s in no_sleep)
    # full jitter draws from [0, ceiling], and the ceiling doubles each attempt
    assert max(no_sleep) <= 16.0


# ---------------------------------------------------------------------------
# Socrata client
# ---------------------------------------------------------------------------


def test_socrata_empty_response_is_an_empty_frame_not_an_error(monkeypatch, tmp_path):
    monkeypatch.setattr(requests, "get", Recorder([FakeResponse(200, "")]))
    df = socrata.fetch_csv("d.test", "abcd-1234", order="id", cache_path=tmp_path / "c.csv")
    assert df.empty


def test_socrata_header_only_response_is_an_empty_frame(monkeypatch, tmp_path):
    monkeypatch.setattr(requests, "get", Recorder([FakeResponse(200, "a,b\n")]))
    df = socrata.fetch_csv("d.test", "abcd-1234", order="id", cache_path=tmp_path / "c.csv")
    assert df.empty


def test_socrata_html_error_page_is_reported_as_source_unavailable(monkeypatch):
    html = "<html><body>1,2,3\n<h1>Gateway Timeout</h1>,,,,,,\n</body></html>"
    monkeypatch.setattr(requests, "get", Recorder([FakeResponse(200, html)]))
    with pytest.raises(SourceUnavailable):
        socrata.fetch_csv("d.test", "abcd-1234", order="id")


def test_socrata_cache_is_reused_without_any_network_call(monkeypatch, tmp_path):
    cache = tmp_path / "cached.csv"
    pd.DataFrame({"id": [1, 2], "v": ["a", "b"]}).to_csv(cache, index=False)

    def explode(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("cache was not used; a network call was made")

    monkeypatch.setattr(requests, "get", explode)
    df = socrata.fetch_csv("d.test", "abcd-1234", order="id", cache_path=cache)
    assert len(df) == 2


def test_socrata_refresh_bypasses_the_cache(monkeypatch, tmp_path):
    cache = tmp_path / "cached.csv"
    pd.DataFrame({"id": [1]}).to_csv(cache, index=False)
    monkeypatch.setattr(requests, "get", Recorder([FakeResponse(200, "id\n7\n8\n")]))
    df = socrata.fetch_csv(
        "d.test", "abcd-1234", order="id", cache_path=cache, refresh=True
    )
    assert df["id"].tolist() == [7, 8]
    assert pd.read_csv(cache)["id"].tolist() == [7, 8]


def test_socrata_paginates_until_a_short_page(monkeypatch, tmp_path):
    full = "id\n" + "\n".join(str(i) for i in range(socrata.PAGE_SIZE))
    rec = Recorder([FakeResponse(200, full + "\n"), FakeResponse(200, "id\n999\n")])
    monkeypatch.setattr(requests, "get", rec)
    df = socrata.fetch_csv("d.test", "abcd-1234", order="id")
    assert len(df) == socrata.PAGE_SIZE + 1
    assert rec.calls[1]["params"]["$offset"] == str(socrata.PAGE_SIZE)


# ---------------------------------------------------------------------------
# ArcGIS client
# ---------------------------------------------------------------------------


def test_arcgis_application_error_is_not_retried(monkeypatch, no_sleep):
    """ArcGIS answers 200 with an `error` object; retrying it is pointless."""
    rec = Recorder([FakeResponse(200, payload={"error": {"code": 400, "message": "x"}})])
    monkeypatch.setattr(requests, "get", rec)
    with pytest.raises(SourceUnavailable):
        arcgis.describe("https://svc.test/FeatureServer/0")
    assert len(rec.calls) == 1


def test_arcgis_non_json_body_is_reported_clearly(monkeypatch, no_sleep):
    monkeypatch.setattr(
        requests, "get", Recorder([FakeResponse(200, "<html>503</html>")])
    )
    with pytest.raises(SourceUnavailable):
        arcgis.describe("https://svc.test/FeatureServer/0")


def test_arcgis_empty_layer_yields_an_empty_frame(monkeypatch):
    monkeypatch.setattr(requests, "get", Recorder([FakeResponse(200, payload={"features": []})]))
    df = arcgis.query("https://svc.test/FeatureServer/0", order_by="OBJECTID")
    assert df.empty


def test_arcgis_features_without_attributes_do_not_crash(monkeypatch):
    payload = {"features": [{"attributes": {"a": 1}}, {"geometry": {}}]}
    monkeypatch.setattr(requests, "get", Recorder([FakeResponse(200, payload=payload)]))
    df = arcgis.query("https://svc.test/FeatureServer/0", order_by="OBJECTID")
    assert len(df) == 2


def test_arcgis_stops_paging_when_the_server_ignores_the_offset(monkeypatch):
    """A server that never advances would otherwise loop for ever."""
    page = {
        "features": [{"attributes": {"i": i}} for i in range(arcgis.PAGE_SIZE)],
        "exceededTransferLimit": True,
    }
    monkeypatch.setattr(requests, "get", Recorder([FakeResponse(200, payload=page)]))
    monkeypatch.setattr(arcgis, "MAX_PAGES", 3)
    with pytest.raises(SourceUnavailable, match="resultOffset"):
        arcgis.query("https://svc.test/FeatureServer/0", order_by="OBJECTID")


def test_arcgis_cache_is_reused_without_any_network_call(monkeypatch, tmp_path):
    cache = tmp_path / "layer.csv"
    pd.DataFrame({"OBJECTID": [1], "issueDate": ["2024-01-01"]}).to_csv(cache, index=False)

    def explode(*a, **k):  # pragma: no cover
        raise AssertionError("cache was not used; a network call was made")

    monkeypatch.setattr(requests, "get", explode)
    df = arcgis.query(
        "https://svc.test/FeatureServer/0",
        order_by="OBJECTID",
        date_fields=("issueDate",),
        cache_path=cache,
    )
    assert len(df) == 1
    assert pd.api.types.is_datetime64_any_dtype(df["issueDate"])


# ---------------------------------------------------------------------------
# per-source isolation in build_dataset
# ---------------------------------------------------------------------------


def _canonical_rows(source: str, n: int = 3) -> pd.DataFrame:
    from msa.config import CANONICAL_COLUMNS

    frame = pd.DataFrame(
        {c: [None] * n for c in CANONICAL_COLUMNS}
    )
    frame["record_id"] = [f"{source}:{i}" for i in range(n)]
    frame["source"] = source
    frame["jurisdiction"] = source.title()
    frame["state"] = "IL"
    frame["application_date"] = pd.Timestamp("2024-01-01")
    frame["issue_date"] = pd.Timestamp("2024-01-10")
    frame["completion_date"] = pd.Timestamp("2024-03-01")
    return frame


def _install_sources(monkeypatch, spec: dict):
    """Replace the source registry with loaders that behave as ``spec`` says."""

    def make(key, behaviour):
        def loader(refresh=False):
            if isinstance(behaviour, Exception):
                raise behaviour
            return behaviour
        return loader

    monkeypatch.setattr(
        build_dataset, "_sources", lambda: {k: make(k, v) for k, v in spec.items()}
    )


def test_one_dead_source_does_not_kill_the_run(monkeypatch):
    _install_sources(
        monkeypatch,
        {
            "alive_a": _canonical_rows("alive_a"),
            "dead": SourceUnavailable("portal is down", source="dead"),
            "alive_b": _canonical_rows("alive_b"),
        },
    )
    df = build_dataset.build()
    assert set(df["source"]) == {"alive_a", "alive_b"}
    assert "dead" in build_dataset.LAST_FAILURES


def test_a_schema_change_in_one_source_is_isolated_too(monkeypatch):
    """This used to abort the whole ingest: the check sat outside the try."""
    broken = _canonical_rows("broken").drop(columns=["issue_date"])
    _install_sources(
        monkeypatch,
        {
            "alive_a": _canonical_rows("alive_a"),
            "broken": broken,
            "alive_b": _canonical_rows("alive_b"),
        },
    )
    df = build_dataset.build()
    assert set(df["source"]) == {"alive_a", "alive_b"}
    assert "issue_date" in build_dataset.LAST_FAILURES["broken"]


def test_an_empty_source_is_isolated(monkeypatch):
    _install_sources(
        monkeypatch,
        {
            "alive_a": _canonical_rows("alive_a"),
            "alive_b": _canonical_rows("alive_b"),
            "empty": _canonical_rows("empty", n=0),
        },
    )
    df = build_dataset.build()
    assert set(df["source"]) == {"alive_a", "alive_b"}
    assert "empty" in build_dataset.LAST_FAILURES


def test_strict_mode_refuses_to_continue(monkeypatch):
    _install_sources(
        monkeypatch,
        {"alive": _canonical_rows("alive"), "dead": SourceUnavailable("down")},
    )
    with pytest.raises(SourceUnavailable):
        build_dataset.build(strict=True)


def test_losing_most_sources_fails_rather_than_shrinking_the_corpus(monkeypatch):
    """A partial corpus silently replacing a complete one is the worst outcome."""
    _install_sources(
        monkeypatch,
        {
            "alive": _canonical_rows("alive"),
            "dead_a": SourceUnavailable("down"),
            "dead_b": SourceUnavailable("down"),
            "dead_c": SourceUnavailable("down"),
        },
    )
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        build_dataset.build()


def test_all_sources_failing_is_an_error(monkeypatch):
    _install_sources(monkeypatch, {"dead": SourceUnavailable("down")})
    with pytest.raises(RuntimeError, match="no source produced data"):
        build_dataset.build()


def test_failures_are_recorded_in_the_dataset_summary(monkeypatch):
    _install_sources(
        monkeypatch,
        {
            "alive_a": _canonical_rows("alive_a"),
            "alive_b": _canonical_rows("alive_b"),
            "dead": SourceUnavailable("portal is down", source="dead"),
        },
    )
    summary = build_dataset.summarise(build_dataset.build())
    assert "dead" in summary["failed_sources"]
    assert summary["total_records"] == 6
