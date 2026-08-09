"""Report what fraction of each segment's durations falls inside its horizon.

The numbers printed here are the ones quoted in ``msa.config`` next to
``MAX_TIMELINE_DAYS_BY_SEGMENT``.

Run::

    python -m scripts.report_horizon_coverage
"""

from __future__ import annotations

import json
import logging

from msa.config import REPORTS_DIR, SEGMENTS, max_timeline_days
from msa.train.data import load_raw


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    raw = load_raw()
    out = {}
    for seg, (a, b) in SEGMENTS.items():
        d = (raw[b] - raw[a]).dt.days.dropna()
        d = d[d >= 0]
        h = max_timeline_days(seg)
        out[seg] = {
            "rows_with_both_dates": len(d),
            "horizon_days": h,
            "coverage_pct": round(float((d <= h).mean() * 100), 2),
            "quantiles": {
                f"p{int(q * 100)}": float(d.quantile(q))
                for q in (0.5, 0.75, 0.9, 0.95, 0.99)
            },
        }
    path = REPORTS_DIR / "horizon_coverage.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
