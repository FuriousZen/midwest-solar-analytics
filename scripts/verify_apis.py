"""Re-run the API introspection checks documented in docs/VERIFIED_APIS.md.

This is the executable half of that document: if a dependency is upgraded and
one of these assumptions stops holding, this script fails loudly instead of the
codebase failing subtly.

Run::

    python -m scripts.verify_apis
"""

from __future__ import annotations

import inspect
import sys


def check(label: str, ok: bool, detail: str = "") -> bool:
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {label}" + (f"  --  {detail}" if detail else ""))
    return ok


def main() -> int:
    ok = True

    import xgboost as xgb

    print(f"\n== xgboost {xgb.__version__} ==")
    params = set(xgb.XGBRegressor().get_params())
    for p in (
        "early_stopping_rounds",
        "enable_categorical",
        "tree_method",
        "max_cat_to_onehot",
        "grow_policy",
        "max_leaves",
        "max_bin",
        "device",
    ):
        ok &= check(f"XGBRegressor accepts `{p}`", p in params)
    fit_args = set(inspect.signature(xgb.XGBRegressor.fit).parameters)
    ok &= check(
        "early_stopping_rounds is NOT a fit() argument",
        "early_stopping_rounds" not in fit_args,
    )
    ok &= check("fit() accepts eval_set", "eval_set" in fit_args)

    import optuna

    print(f"\n== optuna {optuna.__version__} ==")
    cs = inspect.signature(optuna.create_study).parameters
    ok &= check("create_study(direction=...)", "direction" in cs)
    ok &= check("create_study(sampler=...)", "sampler" in cs)
    opt = inspect.signature(optuna.study.Study.optimize).parameters
    ok &= check("Study.optimize(n_trials=...)", "n_trials" in opt)
    tpe = inspect.signature(optuna.samplers.TPESampler.__init__).parameters
    ok &= check("TPESampler(seed=, multivariate=)", {"seed", "multivariate"} <= set(tpe))

    import mlflow

    print(f"\n== mlflow {mlflow.__version__} ==")
    lm = inspect.signature(mlflow.xgboost.log_model).parameters
    ok &= check("xgboost.log_model(name=...) exists", "name" in lm)
    ok &= check("xgboost.log_model(signature=, input_example=)", {"signature", "input_example"} <= set(lm))
    sr = inspect.signature(mlflow.search_runs).parameters
    ok &= check("search_runs(experiment_names=...)", "experiment_names" in sr)

    ok &= check(
        "mlflow.pyfunc.log_model(python_model=, artifacts=)",
        {"python_model", "artifacts"}
        <= set(inspect.signature(mlflow.pyfunc.log_model).parameters),
        "the servable flavour -- the xgboost one cannot round-trip categoricals",
    )

    import shap

    print(f"\n== shap {shap.__version__} ==")
    te = inspect.signature(shap.TreeExplainer.__init__).parameters
    ok &= check("TreeExplainer(model, data=, feature_perturbation=)", {"data", "feature_perturbation"} <= set(te))
    for plot in ("beeswarm", "bar", "waterfall", "scatter"):
        ok &= check(f"shap.plots.{plot} exists", hasattr(shap.plots, plot))

    import sklearn
    from sklearn.model_selection import KFold

    print(f"\n== scikit-learn {sklearn.__version__} ==")
    kf = inspect.signature(KFold.__init__).parameters
    ok &= check("KFold(n_splits, shuffle, random_state)", {"n_splits", "shuffle", "random_state"} <= set(kf))

    import httpx

    print(f"\n== httpx {httpx.__version__} ==")
    # This one is load-bearing: pairing ASGITransport with the *sync* Client
    # raised AttributeError on every invocation of the latency benchmark and
    # the demo script. Assert the asymmetry rather than remember it.
    ok &= check(
        "ASGITransport is async-only (AsyncBaseTransport, no __enter__)",
        issubclass(httpx.ASGITransport, httpx.AsyncBaseTransport)
        and not hasattr(httpx.ASGITransport, "__enter__"),
        "must be driven by httpx.AsyncClient",
    )
    ok &= check(
        "WSGITransport is the synchronous one",
        issubclass(httpx.WSGITransport, httpx.BaseTransport),
    )

    import pydantic

    print(f"\n== pydantic {pydantic.VERSION} ==")
    ok &= check(
        "ConfigDict supports extra / allow_inf_nan / str_max_length",
        {"extra", "allow_inf_nan", "str_max_length"}
        <= set(pydantic.ConfigDict.__annotations__),
    )
    ok &= check(
        "Field(allow_inf_nan=...) is a per-field option",
        "allow_inf_nan" in inspect.signature(pydantic.Field).parameters,
        "without it, JSON Infinity passes a ge=0 bound",
    )

    class _InfCheck(pydantic.BaseModel):
        v: float = pydantic.Field(ge=0, allow_inf_nan=False)

    rejected = False
    try:
        _InfCheck(v=float("inf"))
    except pydantic.ValidationError:
        rejected = True
    ok &= check("allow_inf_nan=False actually rejects +inf", rejected)

    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
