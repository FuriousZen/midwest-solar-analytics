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


#: A run that lost more than this fraction of its registered sources produced
#: a corpus that is not comparable with the one the documented numbers came
#: from, so it fails rather than quietly writing a smaller parquet over the
#: bigger one.
MAX_FAILED_SOURCE_FRACTION = 0.5

#: Populated by the last :func:`build` call; folded into ``dataset_summary.json``
#: so a degraded ingest leaves evidence in a committed artifact.
LAST_FAILURES: dict[str, str] = {}


def build(
    refresh: bool = False,
    only: tuple[str, ...] | None = None,
    *,
    strict: bool = False,
) -> pd.DataFrame:
    """Concatenate every registered source onto the canonical schema.

    Per-source isolation is the whole point of the ``try``: twelve municipal
    portals will not all be up at once, and one dead jurisdiction must cost the
    run that jurisdiction's rows and nothing else. That includes a source whose
    *schema* changed, so the canonical-column check sits inside the isolation,
    not after it; placed outside, a renamed upstream column in one small
    register would abort the entire ingest.

    ``strict=True`` re-raises instead, for CI and for anyone who would rather
    have no dataset than a partial one.
    """
    frames: list[pd.DataFrame] = []
    failures: dict[str, str] = {}
    sources = {k: v for k, v in _sources().items() if not only or k in only}

    for key, loader in sources.items():
        try:
            df = loader(refresh=refresh)
            missing = sorted(set(CANONICAL_COLUMNS) - set(df.columns))
            if missing:
                raise ValueError(
                    f"source {key} is missing canonical columns {missing} "
                    f"(upstream schema change?)"
                )
            if df.empty:
                raise ValueError(f"source {key} returned no rows")
            frame = df[list(CANONICAL_COLUMNS)]
        except Exception as exc:
            if strict:
                raise
            LOG.error("source %s failed, continuing without it: %s", key, exc)
            failures[key] = f"{type(exc).__name__}: {exc}"
            continue
        LOG.info("source %-22s -> %6d rows", key, len(frame))
        frames.append(frame)

    LAST_FAILURES.clear()
    LAST_FAILURES.update(failures)

    if not frames:
        raise RuntimeError(f"no source produced data; failures={failures}")
    if sources and len(failures) / len(sources) > MAX_FAILED_SOURCE_FRACTION:
        raise RuntimeError(
            f"{len(failures)} of {len(sources)} sources failed, which is more "
            f"than the {MAX_FAILED_SOURCE_FRACTION:.0%} tolerated; refusing to "
            f"overwrite the dataset with a partial corpus. failures={failures}"
        )
    if failures:
        LOG.warning(
            "ingest completed with %d of %d sources missing: %s",
            len(failures),
            len(sources),
            sorted(failures),
        )

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
            "records": len(sub),
        }
        for seg, (a, b) in SEGMENTS.items():
            d = (sub[b] - sub[a]).dt.days.dropna()
            d = d[d >= 0]
            entry[seg] = {
                "rows": len(d),
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
        "total_records": len(df),
        "rows_with_usable_duration_by_segment": totals,
        # A source that could not be read is recorded here rather than being
        # silently absent from `sources`; otherwise a degraded ingest looks
        # identical to a complete one that happens to be smaller.
        "failed_sources": dict(LAST_FAILURES),
        "sources": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-download raw sources")
    ap.add_argument("--only", nargs="*", default=None, help="restrict to source keys")
    ap.add_argument(
        "--write",
        action="store_true",
        help="with --only, overwrite the dataset with just those sources",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="fail the run if any single source fails, instead of skipping it",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    only = tuple(args.only) if args.only else None
    df = build(refresh=args.refresh, only=only, strict=args.strict)

    if only and not args.write:
        # Without `--write`, `--only` must not overwrite installations.parquet
        # with a corpus containing just those sources; that would silently
        # shrink the dataset every downstream stage reads. The useful workflow
        # is "refresh one register's cache, then rebuild the whole thing from
        # cache", so that is the default and `--write` is the explicit opt-out.
        print(
            f"--only was given ({', '.join(only)}): fetched {len(df):,} rows and "
            f"refreshed their caches, but NOT overwriting {OUT_PARQUET} with a "
            f"partial corpus. Re-run without --only to rebuild it, or pass "
            f"--write to overwrite deliberately."
        )
        return

    # Write both artifacts atomically: a half-written parquet is exactly the
    # input that makes every downstream stage fail in a confusing way.
    tmp_parquet = OUT_PARQUET.with_suffix(".parquet.tmp")
    df.to_parquet(tmp_parquet, index=False)
    tmp_parquet.replace(OUT_PARQUET)
    summary = summarise(df)
    tmp_summary = OUT_SUMMARY.with_suffix(".json.tmp")
    tmp_summary.write_text(json.dumps(summary, indent=2))
    tmp_summary.replace(OUT_SUMMARY)
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {len(df):,} rows -> {OUT_PARQUET}")
    if summary["failed_sources"]:
        print(
            f"WARNING: {len(summary['failed_sources'])} source(s) failed and are "
            f"absent from this corpus: {sorted(summary['failed_sources'])}"
        )


if __name__ == "__main__":  # pragma: no cover
    main()
