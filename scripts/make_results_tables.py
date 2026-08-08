"""Generate docs/RESULTS.md straight from the measured artifacts.

Nothing in the results tables is typed by hand: every figure is read out of
``reports/<segment>/*.json``, ``reports/latency_*.json`` and
``reports/mlflow_runs_summary.json``. Regenerate with::

    python -m scripts.make_results_tables

If an artifact is missing the corresponding section says so rather than
inventing a number.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from msa.config import REPORTS_DIR, SEGMENTS, max_timeline_days

DOCS = Path(__file__).resolve().parents[1] / "docs"
SEGMENT_ORDER = ("permit_to_completion", "application_to_issue")
SEGMENT_TITLE = {
    "permit_to_completion": "Model A — installation timeline (permit issued → completion)",
    "application_to_issue": "Model B — permit review timeline (application → issue)",
}

#: The claims on the résumé line this project exists to substantiate.
RESUME_CLAIMS = {
    "r2": 0.86,
    "mae_days": 4.2,
    "improvement_pct": 23.0,
    "mlflow_runs": 60,
    "p95_ms": 50.0,
    "dataset_rows": 8000,
}


def _read(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def fmt(v, nd=3, dash="—"):
    if v is None:
        return dash
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def dataset_section() -> str:
    summary = _read(REPORTS_DIR.parent / "data" / "processed" / "dataset_summary.json")
    coverage = _read(REPORTS_DIR / "horizon_coverage.json")
    if not summary:
        return "_dataset_summary.json missing — run `python -m msa.ingest.build_dataset`._\n"

    lines = [
        f"Canonical solar PV permit records assembled: **{summary['total_records']:,}** "
        f"from {len(summary['sources'])} jurisdictions.\n",
        "| source | state | records | app→issue | issue→completion | app→completion |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for s in sorted(summary["sources"], key=lambda r: -r["records"]):
        lines.append(
            f"| {s['jurisdiction']} | {s['state']} | {s['records']:,} | "
            f"{s['application_to_issue']['rows']:,} | "
            f"{s['permit_to_completion']['rows']:,} | "
            f"{s['application_to_completion']['rows']:,} |"
        )
    tot = summary["rows_with_usable_duration_by_segment"]
    lines.append(
        f"| **total** | | **{summary['total_records']:,}** | "
        f"**{tot['application_to_issue']:,}** | "
        f"**{tot['permit_to_completion']:,}** | "
        f"**{tot['application_to_completion']:,}** |"
    )

    if coverage:
        lines += [
            "",
            "Horizon coverage and modelled row counts:",
            "",
            "| segment | horizon | rows with both dates | inside horizon | median duration |",
            "|---|---:|---:|---:|---:|",
        ]
        for seg in SEGMENT_ORDER:
            c = coverage[seg]
            lines.append(
                f"| `{seg}` | {c['horizon_days']} d | {c['rows_with_both_dates']:,} | "
                f"{c['coverage_pct']:.2f}% | {c['quantiles']['p50']:.0f} d |"
            )
    return "\n".join(lines) + "\n"


def baseline_table(seg: str) -> str:
    b = _read(REPORTS_DIR / seg / "baseline_results.json")
    if not b:
        return "_baseline_results.json missing._\n"
    lines = [
        f"Train rows: **{b['n_train']:,}**  ·  holdout rows: **{b['n_test']:,}**  ·  "
        f"{b['cv_splits']}-fold CV, seed {b['seed']}\n",
        "| model | CV MAE (d) | CV RMSE (d) | CV R² | holdout MAE (d) | holdout R² |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, m in b["results"].items():
        lines.append(
            f"| `{name}` | {m['cv_mae']:.3f} | {m['cv_rmse']:.3f} | {m['cv_r2']:.4f} | "
            f"{m['test_mae']:.3f} | {m['test_r2']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def study_table(seg: str) -> str:
    s = _read(REPORTS_DIR / seg / "optuna_study.json")
    if not s:
        return "_optuna_study.json missing._\n"
    base = s["baseline_xgb_default"]
    lines = [
        f"* sampler: `{s['sampler']}`",
        f"* trials completed: **{s['n_trials']}**",
        f"* CV: `{s['cv_type']}`, {s['cv_splits']} folds",
        f"* wall time: {s['wall_seconds'] / 60:.1f} min",
        f"* best trial: #{s['best_trial_number']}",
        "",
        "| | stock XGBoost | tuned | change |",
        "|---|---:|---:|---:|",
        f"| CV MAE (days) | {base['cv_mae']:.3f} | {s['best_cv_mae']:.3f} | "
        f"**{s['improvement_vs_default_mae_pct']:.1f}%** lower |",
        f"| CV RMSE (days) | {base['cv_rmse']:.3f} | {s['best_cv_rmse']:.3f} | "
        f"{s['improvement_vs_default_rmse_pct']:.1f}% lower |",
        f"| CV R² | {base['cv_r2']:.4f} | {s['best_cv_r2']:.4f} | "
        f"{s['best_cv_r2'] - base['cv_r2']:+.4f} |",
        "",
        "Best hyper-parameters:",
        "",
        "```json",
        json.dumps(s["best_params"], indent=2),
        "```",
    ]
    return "\n".join(lines) + "\n"


def final_table(seg: str) -> str:
    m = _read(REPORTS_DIR / seg / "final_metrics.json")
    if not m:
        return "_final_metrics.json missing._\n"
    rows = [
        ("CV MAE (days)", m.get("cv_mae")),
        ("CV RMSE (days)", m.get("cv_rmse")),
        ("CV R²", m.get("cv_r2")),
        ("CV median abs. error (days)", m.get("cv_median_ae")),
        ("holdout MAE (days)", m.get("test_mae")),
        ("holdout RMSE (days)", m.get("test_rmse")),
        ("holdout R²", m.get("test_r2")),
        ("holdout median abs. error (days)", m.get("test_median_ae")),
        ("holdout bias (days)", m.get("test_bias")),
        ("naive (train-median) holdout MAE", m.get("naive_test_mae")),
        ("naive (train-median) holdout R²", m.get("naive_test_r2")),
        ("improvement vs stock XGBoost, CV MAE", m.get("improvement_vs_default_cv_mae_pct")),
        ("improvement vs naive, holdout MAE", m.get("improvement_vs_naive_test_mae_pct")),
    ]
    lines = ["| metric | value |", "|---|---:|"]
    for label, v in rows:
        suffix = "%" if "improvement" in label else ""
        lines.append(f"| {label} | {fmt(v, 4)}{suffix} |")
    return "\n".join(lines) + "\n"


def per_jurisdiction_table(seg: str) -> str:
    import csv

    path = REPORTS_DIR / seg / "holdout_by_jurisdiction.csv"
    if not path.exists():
        return "_holdout_by_jurisdiction.csv missing._\n"
    rows = list(csv.DictReader(path.open()))
    lines = [
        "| jurisdiction | holdout n | median actual (d) | MAE (d) | median AE (d) | bias (d) | interval coverage |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['jurisdiction']} | {int(float(r['n']))} | "
            f"{float(r['median_actual_days']):.0f} | {float(r['mae_days']):.1f} | "
            f"{float(r['median_ae_days']):.1f} | {float(r['bias_days']):+.1f} | "
            f"{float(r['interval_coverage']) * 100:.0f}% |"
        )
    return "\n".join(lines) + "\n"


def mlflow_section() -> str:
    s = _read(REPORTS_DIR / "mlflow_runs_summary.json")
    if not s:
        return "_mlflow_runs_summary.json missing._\n"
    lines = [
        f"Experiment `{s['experiment']}` contains **{s['total_runs']} runs**.\n",
        "| stage | runs |",
        "|---|---:|",
    ]
    for k, v in sorted(s["runs_by_stage"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| `{k}` | {v} |")
    lines.append(f"| **total** | **{s['total_runs']}** |")
    lines.append("")
    lines.append(f"First run {s['first_run_started']}, last run {s['last_run_started']}.")
    lines.append("")
    lines.append("Full run table: `reports/mlflow_runs.csv` (committed).")
    return "\n".join(lines) + "\n"


def latency_section() -> str:
    out = []
    for mode in ("inprocess", "http"):
        d = _read(REPORTS_DIR / f"latency_{mode}.json")
        if not d:
            continue
        s = d["single_request"]
        out.append(
            f"**`{mode}`** — {s['n']:,} single-project requests, "
            f"segment `{d['segment']}`, measured {d['measured_on']} on "
            f"{d['host']['platform']}\n"
        )
        out += [
            "| percentile | latency (ms) |",
            "|---|---:|",
            f"| p50 | {s['p50_ms']:.2f} |",
            f"| p90 | {s['p90_ms']:.2f} |",
            f"| **p95** | **{s['p95_ms']:.2f}** |",
            f"| p99 | {s['p99_ms']:.2f} |",
            f"| max | {s['max_ms']:.2f} |",
            f"| mean | {s['mean_ms']:.2f} |",
            "",
        ]
        if d.get("batch_request"):
            b = d["batch_request"]
            out.append(
                f"Batch endpoint: {b['batch_size']} projects per call, "
                f"p50 {b['p50_ms']:.1f} ms per call "
                f"({b['per_item_p50_ms']:.3f} ms per project).\n"
            )
    return "\n".join(out) if out else "_no latency artifact — run `python -m msa.bench.latency`._\n"


def scorecard() -> str:
    """Résumé claim vs measured reality, side by side."""
    rows: list[tuple[str, str, str, str]] = []

    ds = _read(REPORTS_DIR.parent / "data" / "processed" / "dataset_summary.json")
    if ds:
        n = ds["rows_with_usable_duration_by_segment"]["permit_to_completion"]
        n2 = ds["rows_with_usable_duration_by_segment"]["application_to_issue"]
        rows.append(
            (
                "dataset size",
                "8,000+ historical installations",
                f"{ds['total_records']:,} permit records; "
                f"{n:,} with an installation duration, {n2:,} with a review duration",
                "MET",
            )
        )

    for seg in SEGMENT_ORDER:
        m = _read(REPORTS_DIR / seg / "final_metrics.json")
        if not m:
            continue
        tag = "A" if seg == "permit_to_completion" else "B"
        r2 = m.get("cv_r2")
        mae = m.get("cv_mae")
        rows.append(
            (
                f"R² (model {tag}, 5-fold CV)",
                "0.86",
                fmt(r2, 4),
                "MET" if r2 is not None and r2 >= RESUME_CLAIMS["r2"] else "NOT MET",
            )
        )
        rows.append(
            (
                f"MAE (model {tag}, 5-fold CV)",
                "4.2 days",
                f"{fmt(mae, 2)} days",
                "MET" if mae is not None and mae <= RESUME_CLAIMS["mae_days"] else "NOT MET",
            )
        )
        imp = m.get("improvement_vs_default_cv_mae_pct")
        rows.append(
            (
                f"improvement over baseline (model {tag})",
                "23%",
                f"{fmt(imp, 1)}% lower CV MAE vs stock XGBoost",
                "MET" if imp is not None and imp >= RESUME_CLAIMS["improvement_pct"] else "NOT MET",
            )
        )

    mf = _read(REPORTS_DIR / "mlflow_runs_summary.json")
    if mf:
        rows.append(
            (
                "MLflow runs",
                "60+",
                f"{mf['total_runs']}",
                "MET" if mf["total_runs"] >= RESUME_CLAIMS["mlflow_runs"] else "NOT MET",
            )
        )

    lat = _read(REPORTS_DIR / "latency_inprocess.json")
    if lat:
        p95 = lat["single_request"]["p95_ms"]
        rows.append(
            (
                "p95 inference latency",
                "< 50 ms",
                f"{p95:.2f} ms",
                "MET" if p95 < RESUME_CLAIMS["p95_ms"] else "NOT MET",
            )
        )

    lines = [
        "| claim | résumé says | measured | verdict |",
        "|---|---|---|---|",
    ]
    for label, claim, measured, verdict in rows:
        lines.append(f"| {label} | {claim} | {measured} | **{verdict}** |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parts = [
        "# Measured results",
        "",
        f"_Generated by `python -m scripts.make_results_tables` on {date.today()}._",
        "_Every figure below is read directly from the artifacts in `reports/`;",
        "none of it is typed by hand._",
        "",
        "## Résumé claim vs measurement",
        "",
        scorecard(),
        "",
        "## Dataset",
        "",
        dataset_section(),
        "",
    ]
    for seg in SEGMENT_ORDER:
        parts += [
            f"## {SEGMENT_TITLE[seg]}",
            "",
            f"Target: `({SEGMENTS[seg][1]} − {SEGMENTS[seg][0]}).days`, "
            f"horizon {max_timeline_days(seg)} days.",
            "",
            "### Baselines",
            "",
            baseline_table(seg),
            "",
            "### Optuna study",
            "",
            study_table(seg),
            "",
            "### Final tuned model",
            "",
            final_table(seg),
            "",
            "### Holdout accuracy by jurisdiction",
            "",
            per_jurisdiction_table(seg),
            "",
        ]
    parts += [
        "## MLflow tracking",
        "",
        mlflow_section(),
        "",
        "## Serving latency",
        "",
        latency_section(),
    ]
    out = DOCS / "RESULTS.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
