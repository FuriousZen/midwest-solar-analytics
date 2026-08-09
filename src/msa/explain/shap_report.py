"""SHAP attribution dashboard for the trained timeline model.

Produces, per segment, under ``reports/<segment>/``:

* ``shap_beeswarm.png``   global attribution distribution, one row per feature
* ``shap_bar.png``        mean |SHAP| ranking
* ``shap_waterfall_*.png`` three worked single-prediction explanations
* ``shap_dependence_*.png`` dependence plots for the top drivers
* ``shap_values.parquet`` the raw per-row attributions, so a stakeholder can
  join them back to their own permits
* ``shap_dashboard.html`` a self-contained page tying it together, including a
  per-jurisdiction breakdown of which drivers matter where

Exactness note: ``shap.TreeExplainer`` on a gradient-boosted tree computes exact
Shapley values via TreeSHAP, so these are attributions, not approximations.
Attributions are in the model's output space -- if the model was fitted on
``log1p(days)``, SHAP values are in log-days and the dashboard says so.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from msa.config import DEFAULT_SEGMENT, MODELS_DIR, REPORTS_DIR, SEGMENTS
from msa.train.data import load_dataset

LOG = logging.getLogger(__name__)

MAX_EXPLAIN_ROWS = 4000


def load_model(segment: str) -> tuple[xgb.XGBRegressor, dict]:
    md = MODELS_DIR / segment
    meta = json.loads((md / "metadata.json").read_text())
    model = xgb.XGBRegressor()
    model.load_model(str(md / "model.json"))
    return model, meta


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _save(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    b64 = _fig_to_b64(fig)
    return b64


def compute(segment: str, max_rows: int = MAX_EXPLAIN_ROWS):
    model, meta = load_model(segment)
    ds = load_dataset(segment=segment)

    X = ds.X_test if len(ds.X_test) else ds.X_train
    metaframe = ds.meta_test if len(ds.X_test) else ds.meta_train
    y = ds.y_test if len(ds.X_test) else ds.y_train
    if len(X) > max_rows:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(len(X), max_rows, replace=False))
        X, metaframe, y = X.iloc[idx], metaframe.iloc[idx], y.iloc[idx]

    explainer = shap.TreeExplainer(model)
    explanation = explainer(X)
    LOG.info(
        "SHAP: %d rows x %d features, base value %.3f",
        *explanation.values.shape,
        float(np.ravel(explanation.base_values)[0]),
    )
    return model, meta, ds, X, metaframe.reset_index(drop=True), y.reset_index(drop=True), explanation


def build(segment: str, max_rows: int = MAX_EXPLAIN_ROWS) -> Path:
    out = REPORTS_DIR / segment
    out.mkdir(parents=True, exist_ok=True)
    model, meta, _ds, X, metaframe, y, expl = compute(segment, max_rows)
    log_target = bool(meta.get("log1p_target"))
    unit = "log1p(days)" if log_target else "days"

    images: dict[str, str] = {}

    plt.figure(figsize=(9, 8))
    shap.plots.beeswarm(expl, max_display=20, show=False)
    plt.title(f"SHAP attributions -- {segment} ({unit})", fontsize=11)
    images["beeswarm"] = _save(plt.gcf(), out / "shap_beeswarm.png")

    plt.figure(figsize=(9, 7))
    shap.plots.bar(expl, max_display=20, show=False)
    plt.title(f"Mean |SHAP| -- {segment} ({unit})", fontsize=11)
    images["bar"] = _save(plt.gcf(), out / "shap_bar.png")

    mean_abs = pd.Series(
        np.abs(expl.values).mean(axis=0), index=list(X.columns)
    ).sort_values(ascending=False)
    top = mean_abs.index[:4].tolist()

    for name in top[:3]:
        try:
            plt.figure(figsize=(7, 5))
            shap.plots.scatter(expl[:, name], show=False)
            plt.title(f"{name} -- attribution vs value", fontsize=11)
            images[f"dep_{name}"] = _save(
                plt.gcf(), out / f"shap_dependence_{name}.png"
            )
        except Exception as exc:
            LOG.warning("dependence plot for %s skipped: %s", name, exc)
            plt.close("all")

    # three worked examples: a fast job, a typical job, a slow job
    order = np.argsort(y.to_numpy())
    picks = {
        "fastest_decile": int(order[len(order) // 10]),
        "median": int(order[len(order) // 2]),
        "slowest_decile": int(order[-max(1, len(order) // 10)]),
    }
    examples = []
    for label, i in picks.items():
        plt.figure(figsize=(9, 6))
        shap.plots.waterfall(expl[i], max_display=14, show=False)
        plt.title(f"{label}: {metaframe.loc[i, 'jurisdiction']}", fontsize=11)
        images[f"wf_{label}"] = _save(plt.gcf(), out / f"shap_waterfall_{label}.png")
        pred = float(model.predict(X.iloc[[i]])[0])
        if log_target:
            pred = float(np.expm1(pred))
        examples.append(
            {
                "label": label,
                "record_id": metaframe.loc[i, "record_id"],
                "jurisdiction": metaframe.loc[i, "jurisdiction"],
                "start_date": str(pd.Timestamp(metaframe.loc[i, "start_date"]).date()),
                "actual_days": float(y.iloc[i]),
                "predicted_days": max(pred, 0.0),
                "image": f"wf_{label}",
            }
        )

    values = pd.DataFrame(expl.values, columns=list(X.columns))
    values.insert(0, "record_id", metaframe["record_id"].to_numpy())
    values.insert(1, "base_value", np.ravel(expl.base_values))
    values.to_parquet(out / "shap_values.parquet", index=False)

    # per-jurisdiction driver ranking
    by_juris = {}
    for juris, idx in metaframe.groupby("jurisdiction").groups.items():
        pos = metaframe.index.get_indexer(idx)
        sub = np.abs(expl.values[pos]).mean(axis=0)
        s = pd.Series(sub, index=list(X.columns)).sort_values(ascending=False)
        by_juris[juris] = {
            "n": len(pos),
            "top_drivers": [
                {"feature": f, "mean_abs_shap": float(v)} for f, v in s.head(6).items()
            ],
        }

    summary = {
        "segment": segment,
        "unit": unit,
        "n_explained": len(X),
        "base_value": float(np.ravel(expl.base_values)[0]),
        "global_ranking": [
            {"feature": f, "mean_abs_shap": float(v)} for f, v in mean_abs.items()
        ],
        "by_jurisdiction": by_juris,
        "examples": examples,
    }
    (out / "shap_summary.json").write_text(json.dumps(summary, indent=2))

    html = render_html(summary, images, meta)
    path = out / "shap_dashboard.html"
    path.write_text(html)
    LOG.info("wrote %s", path)
    return path


def render_html(summary: dict, images: dict[str, str], meta: dict) -> str:
    seg = summary["segment"]
    start, end = SEGMENTS[seg]

    def img(key: str, alt: str) -> str:
        if key not in images:
            return f"<p><em>({alt} unavailable)</em></p>"
        return f'<img alt="{alt}" src="data:image/png;base64,{images[key]}">'

    rows = "".join(
        f"<tr><td>{i + 1}</td><td><code>{d['feature']}</code></td>"
        f"<td class='num'>{d['mean_abs_shap']:.4f}</td></tr>"
        for i, d in enumerate(summary["global_ranking"][:25])
    )

    juris_blocks = "".join(
        "<div class='card'><h4>{j} <span class='muted'>n={n}</span></h4><ol>{items}</ol></div>".format(
            j=j,
            n=v["n"],
            items="".join(
                f"<li><code>{d['feature']}</code> "
                f"<span class='muted'>{d['mean_abs_shap']:.3f}</span></li>"
                for d in v["top_drivers"]
            ),
        )
        for j, v in sorted(
            summary["by_jurisdiction"].items(), key=lambda kv: -kv[1]["n"]
        )
    )

    example_blocks = "".join(
        "<div class='example'><h4>{label}</h4>"
        "<p class='muted'>{jur} &middot; started {d} &middot; "
        "actual <b>{a:.0f} d</b> &middot; predicted <b>{p:.1f} d</b></p>{im}</div>".format(
            label=e["label"].replace("_", " "),
            jur=e["jurisdiction"],
            d=e["start_date"],
            a=e["actual_days"],
            p=e["predicted_days"],
            im=img(e["image"], e["label"]),
        )
        for e in summary["examples"]
    )

    m = meta.get("metrics", {})
    metric_rows = "".join(
        f"<tr><td>{k}</td><td class='num'>{v:.4f}</td></tr>"
        for k, v in m.items()
        if isinstance(v, (int, float)) and v is not None
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SHAP dashboard - {seg}</title>
<style>
  :root {{ color-scheme: light dark; --fg:#16181d; --bg:#fff; --muted:#6b7280;
           --line:#e5e7eb; --card:#f9fafb; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --fg:#e6e8eb; --bg:#101216; --muted:#9aa1ab; --line:#272b33; --card:#171a20; }}
  }}
  body {{ margin:0; padding:2rem 1.25rem 4rem; background:var(--bg); color:var(--fg);
          font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
  main {{ max-width: 1080px; margin: 0 auto; }}
  h1 {{ font-size:1.6rem; margin:0 0 .25rem; }}
  h2 {{ font-size:1.15rem; margin:2.5rem 0 .75rem; padding-bottom:.35rem;
        border-bottom:1px solid var(--line); }}
  h4 {{ margin:0 0 .35rem; font-size:.95rem; }}
  .muted {{ color:var(--muted); font-weight:400; }}
  img {{ max-width:100%; height:auto; display:block; margin:.5rem 0 1.5rem;
         background:#fff; border-radius:8px; }}
  table {{ border-collapse:collapse; width:100%; font-size:.9rem; }}
  th,td {{ text-align:left; padding:.35rem .6rem; border-bottom:1px solid var(--line); }}
  td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  code {{ font-size:.85em; }}
  .scroll {{ overflow-x:auto; }}
  .grid {{ display:grid; gap:1rem; grid-template-columns:repeat(auto-fill,minmax(240px,1fr)); }}
  .card, .example {{ background:var(--card); border:1px solid var(--line);
                     border-radius:10px; padding:.85rem 1rem; }}
  ol {{ margin:.25rem 0 0 1.1rem; padding:0; }}
  .lead {{ color:var(--muted); margin:.25rem 0 0; }}
</style></head><body><main>

<h1>Per-feature attributions &mdash; {seg}</h1>
<p class="lead">Target: <code>({end} &minus; {start}).days</code>.
Attribution unit: <b>{summary["unit"]}</b>.
Explained rows: {summary["n_explained"]} (held-out set).
Model base value: {summary["base_value"]:.3f}.</p>

<h2>What moves the prediction, overall</h2>
{img("beeswarm", "SHAP beeswarm")}
{img("bar", "mean absolute SHAP")}

<h2>Ranking</h2>
<div class="scroll"><table>
<thead><tr><th>#</th><th>feature</th><th>mean |SHAP|</th></tr></thead>
<tbody>{rows}</tbody></table></div>

<h2>Worked examples</h2>
{example_blocks}

<h2>Drivers by jurisdiction</h2>
<div class="grid">{juris_blocks}</div>

<h2>Model metrics</h2>
<div class="scroll"><table>
<thead><tr><th>metric</th><th>value</th></tr></thead>
<tbody>{metric_rows}</tbody></table></div>

<p class="lead">Generated by <code>python -m msa.explain.shap_report --segment {seg}</code>.
Raw attributions: <code>reports/{seg}/shap_values.parquet</code>.</p>
</main></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment", default=DEFAULT_SEGMENT, choices=list(SEGMENTS))
    ap.add_argument("--max-rows", type=int, default=MAX_EXPLAIN_ROWS)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(build(args.segment, args.max_rows))


if __name__ == "__main__":  # pragma: no cover
    main()
