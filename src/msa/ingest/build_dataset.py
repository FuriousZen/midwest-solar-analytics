"""Assemble every ingester into one canonical installation-timeline dataset.

Run::

    python -m msa.ingest.build_dataset            # uses cached raw downloads
    python -m msa.ingest.build_dataset --refresh  # re-download from source
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable

import pandas as pd

from msa.config import CANONICAL_COLUMNS, PROCESSED_DIR, SEGMENTS

LOG = logging.getLogger(__name__)


#: Module names inside ``msa.ingest`` that expose ``SOURCE_KEY`` and ``load()``.
SOURCE_MODULES: tuple[str, ...] = (
    "chicago",
    "minneapolis",
    "saintpaul",
    "goldenvalley",
    "kansascity",
    "cincinnati",
)


def _sources() -> dict[str, Callable[..., pd.DataFrame]]:
    """Every registered ingester, keyed by its canonical ``source`` value."""
    import importlib

    out: dict[str, Callable[..., pd.DataFrame]] = {}
    for name in SOURCE_MODULES:
        mod = importlib.import_module(f"msa.ingest.{name}")
        out[mod.SOURCE_KEY] = mod.load

    # the small ArcGIS registers share one declarative module
    from msa.ingest import arcgis_permits

    for key in arcgis_permits.LAYERS:
        out[key] = arcgis_permits.make_module_api(key)[1]
    return out


OUT_PARQUET = PROCESSED_DIR / "installations.parquet"
OUT_SUMMARY = PROCESSED_DIR / "dataset_summary.json"


def build(refresh: bool = False, only: tuple[str, ...] | None = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    failures: dict[str, str] = {}
    for key, loader in _sources().items():
        if only and key not in only:
            continue
        try:
            df = loader(refresh=refresh)
        except Exception as exc:  # noqa: BLE001 - one bad portal must not kill the run
            LOG.error("source %s failed: %s", key, exc)
            failures[key] = str(exc)
            continue
        missing = set(CANONICAL_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"source {key} is missing canonical columns {missing}")
        LOG.info("source %-22s -> %6d rows", key, len(df))
        frames.append(df[list(CANONICAL_COLUMNS)])

    if not frames:
        raise RuntimeError(f"no source produced data; failures={failures}")

    out = pd.concat(frames, ignore_index=True)
    for c in ("application_date", "issue_date", "completion_date"):
        out[c] = pd.to_datetime(out[c], errors="coerce")
    out = out.drop_duplicates(subset=["record_id"]).reset_index(drop=True)
    return out


def summarise(df: pd.DataFrame) -> dict:
    """Row counts and observed durations per source and per milestone segment."""
    rows = []
    for (src, juris, state), sub in df.groupby(
        ["source", "jurisdiction", "state"], sort=True
    ):
        entry = {
            "source": src,
            "jurisdiction": juris,
            "state": state,
            "records": int(len(sub)),
        }
        for seg, (a, b) in SEGMENTS.items():
            d = (sub[b] - sub[a]).dt.days.dropna()
            d = d[d >= 0]
            entry[seg] = {
                "rows": int(len(d)),
                "median_days": float(d.median()) if len(d) else None,
                "mean_days": float(d.mean()) if len(d) else None,
                "p90_days": float(d.quantile(0.9)) if len(d) else None,
            }
        for c, label in (
            ("application_date", "application"),
            ("issue_date", "issue"),
            ("completion_date", "completion"),
        ):
            s = sub[c].dropna()
            entry[f"{label}_date_range"] = (
                [str(s.min().date()), str(s.max().date())] if len(s) else None
            )
        rows.append(entry)

    totals = {}
    for seg, (a, b) in SEGMENTS.items():
        d = (df[b] - df[a]).dt.days.dropna()
        totals[seg] = int((d >= 0).sum())
    return {
        "total_records": int(len(df)),
        "rows_with_usable_duration_by_segment": totals,
        "sources": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-download raw sources")
    ap.add_argument("--only", nargs="*", default=None, help="restrict to source keys")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    df = build(refresh=args.refresh, only=tuple(args.only) if args.only else None)
    df.to_parquet(OUT_PARQUET, index=False)
    summary = summarise(df)
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {len(df):,} rows -> {OUT_PARQUET}")


if __name__ == "__main__":  # pragma: no cover
    main()
