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
import os
from datetime import date
from pathlib import Path

from msa.config import REPORTS_DIR, SEGMENTS, max_timeline_days

#: Where RESULTS.md is written. Overridable so the CI smoke pipeline can
#: generate its tables into a scratch directory instead of over the committed
#: docs -- a smoke run must never rewrite the numbers a real run measured.
DOCS = Path(os.environ.get("MSA_DOCS_DIR", Path(__file__).resolve().parents[1] / "docs"))
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
        f"* trials completed: **{s.get('n_trials_completed', s['n_trials'])}**"
        + (
            f" (failed: {s['n_trials_failed']}, total recorded: {s['n_trials']})"
            if s.get("n_trials_failed")
            else ""
        ),
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
        ("80% interval empirical coverage (holdout)", m.get("interval_empirical_coverage_test")),
        ("80% interval coverage before conformal calibration", m.get("interval_uncalibrated_coverage_test")),
        ("80% interval median width (days)", m.get("interval_median_width_days")),
    ]
    lines = ["| metric | value |", "|---|---:|"]
    for label, v in rows:
        lines.append(f"| {label} | {fmt(v, 4)} |")

    lines += [
        "",
        "Reduction in 5-fold CV MAE achieved by the tuned model, against each",
        "baseline measured on the identical folds:",
        "",
        "| baseline | baseline CV MAE (d) | tuned CV MAE (d) | reduction |",
        "|---|---:|---:|---:|",
    ]
    b = _read(REPORTS_DIR / seg / "baseline_results.json")
    tuned = m.get("cv_mae")
    if b and tuned is not None:
        for name, bm in b["results"].items():
            key = f"improvement_vs_{name}_cv_mae_pct"
            imp = m.get(key)
            lines.append(
                f"| `{name}` | {bm['cv_mae']:.3f} | {tuned:.3f} | "
                f"{fmt(imp, 1)}% |"
            )
    return "\n".join(lines) + "\n"


def per_jurisdiction_table(seg: str) -> str:
    import csv

    path = REPORTS_DIR / seg / "holdout_by_jurisdiction.csv"
    if not path.exists():
        return "_holdout_by_jurisdiction.csv missing._\n"
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
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
        for seg in SEGMENT_ORDER:
            d = _read(REPORTS_DIR / f"latency_{mode}_{seg}.json")
            if not d:
                continue
            out.extend(_latency_block(d, mode))
    return "\n".join(out) if out else "_no latency artifact — run `python -m msa.bench.latency`._\n"


def _latency_block(d: dict, mode: str) -> list[str]:
    s = d["single_request"]
    out = [
        f"**`{mode}` · `{d['segment']}`** — {s['n']:,} single-project requests, "
        f"measured {d['measured_on']} on {d['host']['platform']}\n",
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
    by_conc = d.get("by_concurrency") or {}
    if len(by_conc) > 1:
        out += [
            "Under concurrent load (same request mix, N in flight at once):",
            "",
            "| concurrent requests | p50 (ms) | p95 (ms) | throughput (req/s) |",
            "|---:|---:|---:|---:|",
        ]
        for level in sorted(by_conc, key=int):
            c = by_conc[level]
            out.append(
                f"| {level} | {c['p50_ms']:.2f} | {c['p95_ms']:.2f} | "
                f"{c['throughput_rps']:.0f} |"
            )
        out.append("")
    if d.get("batch_request"):
        b = d["batch_request"]
        out.append(
            f"Batch endpoint: {b['batch_size']} projects per call, "
            f"p50 {b['p50_ms']:.1f} ms per call "
            f"({b['per_item_p50_ms']:.3f} ms per project).\n"
        )
    return out


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

    # A claim whose artifact is absent must be reported as NOT MEASURED, not
    # silently dropped from the table. Omission reads as "fine" to a skimmer;
    # that is exactly the failure mode this scorecard exists to prevent.
    def unmeasured(cmd: str) -> tuple[str, str]:
        return (f"not measured — run `{cmd}`", "NOT MEASURED")

    for seg in SEGMENT_ORDER:
        tag = "A" if seg == "permit_to_completion" else "B"
        m = _read(REPORTS_DIR / seg / "final_metrics.json")
        cmd = f"make tune final SEGMENT={seg}"

        if not m:
            for label, claim in (
                (f"R² (model {tag}, 5-fold CV)", "0.86"),
                (f"MAE (model {tag}, 5-fold CV)", "4.2 days"),
                (f"improvement over baseline (model {tag})", "23%"),
            ):
                rows.append((label, claim, *unmeasured(cmd)))
            continue

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
        imp = m.get("improvement_vs_xgb_default_cv_mae_pct") or m.get(
            "improvement_vs_default_cv_mae_pct"
        )
        naive_imp = m.get("improvement_vs_global_median_cv_mae_pct")
        extra = (
            f" (vs the naive median baseline: {naive_imp:.1f}%)"
            if naive_imp is not None
            else ""
        )
        rows.append(
            (
                f"improvement over baseline (model {tag})",
                "23%",
                f"{fmt(imp, 1)}% lower CV MAE vs stock XGBoost{extra}",
                "MET" if imp is not None and imp >= RESUME_CLAIMS["improvement_pct"] else "NOT MET",
            )
        )

    mf = _read(REPORTS_DIR / "mlflow_runs_summary.json")
    if mf:
        by_stage = mf.get("runs_by_stage", {})
        detail = ", ".join(f"{v} {k}" for k, v in sorted(by_stage.items(), key=lambda kv: -kv[1]))
        rows.append(
            (
                "MLflow runs",
                "60+",
                f"{mf['total_runs']} ({detail})",
                "MET" if mf["total_runs"] >= RESUME_CLAIMS["mlflow_runs"] else "NOT MET",
            )
        )
    else:
        rows.append(
            ("MLflow runs", "60+", *unmeasured("python -m scripts.export_mlflow_summary"))
        )

    lat = _read(REPORTS_DIR / "latency_inprocess_permit_to_completion.json")
    rows.append(
        (
            "p95 inference latency",
            "< 50 ms",
            *(
                (
                    f"{lat['single_request']['p95_ms']:.2f} ms",
                    "MET"
                    if lat["single_request"]["p95_ms"] < RESUME_CLAIMS["p95_ms"]
                    else "NOT MET",
                )
                if lat
                else unmeasured("make bench")
            ),
        )
    )

    lines = [
        "| claim | résumé says | measured | verdict |",
        "|---|---|---|---|",
    ]
    for label, claim, measured, verdict in rows:
        lines.append(f"| {label} | {claim} | {measured} | **{verdict}** |")
    return "\n".join(lines) + "\n"


def partial_study_section() -> str:
    """Report an Optuna study that was logged but never completed.

    Discarding 136 real trials because the run was interrupted would waste
    genuine evidence; presenting them as final results would be a lie. So they
    are reported, loudly labelled INCOMPLETE, and the derived improvement is
    computed against the same baseline the finished study would use.
    """
    mf = _read(REPORTS_DIR / "mlflow_runs_summary.json")
    if not mf or not mf.get("top_trials_by_cv_mae"):
        return ""
    out: list[str] = []
    for seg in SEGMENT_ORDER:
        if _read(REPORTS_DIR / seg / "optuna_study.json"):
            continue  # a completed study is reported properly elsewhere
        trials = [t for t in mf["top_trials_by_cv_mae"] if t.get("segment") == seg]
        if not trials:
            continue
        n = mf.get("runs_by_segment_and_stage", {}).get(seg, {}).get("optuna", 0)
        best = min(trials, key=lambda t: t["cv_mae"])
        bl = _read(REPORTS_DIR / seg / "baseline_results.json")
        out += [
            f"#### `{seg}` — INCOMPLETE study, {n} of 200 trials logged",
            "",
            "> These are real logged trials from a study that was **stopped before",
            "> completing**. No final model was fitted from them and no holdout was",
            "> scored, so they are *not* the project's results — they are a partial",
            "> read on where a finished study would land.",
            "",
            "| | value |",
            "|---|---:|",
            f"| trials logged | {n} of 200 |",
            f"| best trial CV MAE (days) | {best['cv_mae']:.3f} |",
            f"| best trial CV R² | {best['cv_r2']:.4f} |",
        ]
        if bl:
            d = bl["results"].get("xgb_default")
            if d:
                imp = 100.0 * (d["cv_mae"] - best["cv_mae"]) / d["cv_mae"]
                out += [
                    f"| stock XGBoost CV MAE (days) | {d['cv_mae']:.3f} |",
                    f"| reduction in CV MAE so far | {imp:.1f}% |",
                ]
        out.append("")
    return "\n".join(out)


README_START = "<!-- RESULTS_TABLE -->"


def readme_block() -> str:
    """The headline block injected into README.md between the markers."""
    parts = [
        README_START,
        "",
        "### Résumé claim vs measurement",
        "",
        scorecard(),
        "",
    ]

    a = _read(REPORTS_DIR / "permit_to_completion" / "final_metrics.json") or {}
    b = _read(REPORTS_DIR / "application_to_issue" / "final_metrics.json") or {}
    sa = _read(REPORTS_DIR / "permit_to_completion" / "optuna_study.json") or {}
    sb = _read(REPORTS_DIR / "application_to_issue" / "optuna_study.json") or {}

    if a or b:
        parts += [
            "### Headline metrics (tuned models)",
            "",
            "The trial budget each model actually got is the first row, because",
            "every metric below is conditional on it. Neither study ran to 200",
            "trials: model A's was stopped at 136 and the final model was fitted",
            "from those; model B's was deliberately run short.",
            "",
            "| | Model A — installation timeline | Model B — permit review |",
            "|---|---:|---:|",
            f"| Optuna trials completed | {sa.get('n_trials_completed', 'not measured')} | "
            f"{sb.get('n_trials_completed', 'not measured')} |",
        ]

        def row(label, key, nd=3, pct=False):
            def f(v):
                if v is None:
                    return "not measured"
                return f"{v:.{nd}f}" + ("%" if pct else "")

            parts.append(f"| {label} | {f(a.get(key))} | {f(b.get(key))} |")

        row("training rows", "n_train")
        row("5-fold CV MAE (days)", "cv_mae", 2)
        row("5-fold CV RMSE (days)", "cv_rmse", 2)
        row("5-fold CV R²", "cv_r2", 4)
        row("chronological holdout MAE (days)", "test_mae", 2)
        row("chronological holdout R²", "test_r2", 4)
        row("improvement vs stock XGBoost (CV MAE)", "improvement_vs_xgb_default_cv_mae_pct", 1, True)
        row("80% interval coverage on holdout", "interval_empirical_coverage_test", 3)
    else:
        parts += [
            "### Headline metrics (tuned models)",
            "",
            "> **Not measured.** The 200-trial Optuna study was stopped before",
            "> completing, so no tuned model was fitted, no holdout was scored,",
            "> and there is no latency distribution or SHAP dashboard. Rather",
            "> than print placeholder numbers this section stays empty until",
            "> `./scripts/run_pipeline.sh` has produced the artifacts, at which",
            "> point `python -m scripts.make_results_tables` fills it in.",
            "",
            "What **is** measured is the baseline layer below — four models per",
            "segment on identical 5-fold splits, which is what the tuned model",
            "must be judged against.",
        ]

    ps = partial_study_section()
    if ps:
        parts += ["", "### Partial Optuna study (incomplete)", "", ps]

    parts += [
        "",
        "### Measured baselines",
        "",
    ]
    for seg in SEGMENT_ORDER:
        bl = _read(REPORTS_DIR / seg / "baseline_results.json")
        if not bl:
            continue
        parts += [
            f"`{seg}` — {bl['n_train']:,} train / {bl['n_test']:,} holdout, "
            f"{bl['cv_splits']}-fold CV, seed {bl['seed']}",
            "",
            "| model | CV MAE (d) | CV R² | holdout MAE (d) | holdout R² |",
            "|---|---:|---:|---:|---:|",
        ]
        for name, m in bl["results"].items():
            parts.append(
                f"| `{name}` | {m['cv_mae']:.3f} | {m['cv_r2']:.4f} | "
                f"{m['test_mae']:.3f} | {m['test_r2']:.4f} |"
            )
        parts.append("")

    parts += [
        "Full tables, per-jurisdiction breakdowns, the Optuna study and the",
        "latency distribution: **[docs/RESULTS.md](docs/RESULTS.md)**.",
        "",
    ]
    return "\n".join(parts)


def inject_readme() -> None:
    if os.environ.get("MSA_DOCS_DIR"):
        print("MSA_DOCS_DIR is set; leaving README.md alone")
        return
    path = Path(__file__).resolve().parents[1] / "README.md"
    text = path.read_text()
    if README_START not in text:
        print("README marker not found; skipping injection")
        return
    head, _, tail = text.partition(README_START)
    # everything up to the next '---' separator belongs to the generated block
    _, sep, rest = tail.partition("\n---\n")
    path.write_text(head + readme_block() + "\n" + sep.lstrip("\n") + rest)
    print(f"injected results block into {path}")


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
        "## Partial Optuna study",
        "",
        partial_study_section() or "_no partial study logged._\n",
        "",
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
    inject_readme()


if __name__ == "__main__":
    main()
