"""The training side of the artifact contract, and the MLflow flavour.

Two things are checked here that nothing else covers:

1. Everything ``msa.train.final`` writes is loadable by ``msa.serve`` -- the
   round trip, not each half in isolation.
2. The model logged to MLflow can actually be *served*. The xgboost flavour
   could not: MLflow serves a DataFrame as JSON, JSON has no categorical
   dtype, and the booster rejects the ``object`` columns it gets back. MLflow
   only warns about that at log time, so without a test the registry quietly
   holds an unusable model.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from msa.artifacts import ARTIFACT_SCHEMA_VERSION, REQUIRED_METADATA_KEYS
from msa.features.build import CATEGORICAL_FEATURES, FEATURES
from msa.serve.app import ModelBundle

SEGMENT = "permit_to_completion"


def test_written_metadata_satisfies_the_serving_contract(model_root):
    meta = json.loads((model_root / SEGMENT / "metadata.json").read_text())
    for key in REQUIRED_METADATA_KEYS:
        assert key in meta, key
    assert meta["artifact_schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert list(meta["features"]) == list(FEATURES)
    assert set(meta["categories"]) >= set(CATEGORICAL_FEATURES)


def test_final_stamps_the_schema_version_and_a_content_bound_version():
    """`build_metadata` must not be able to emit an unstamped artifact."""
    import inspect

    from msa.train import final

    source = inspect.getsource(final.build_metadata)
    assert "artifact_schema_version" in source
    assert "trained_at" in source
    assert "build_model_version" in inspect.getsource(final.run)


def test_serving_can_load_what_training_wrote(model_root):
    bundle = ModelBundle(SEGMENT, root=model_root)
    assert bundle.artifact_schema_version == ARTIFACT_SCHEMA_VERSION
    assert bundle.version
    assert np.isfinite(bundle.smoke_test())


# ---------------------------------------------------------------------------
# the MLflow flavour
# ---------------------------------------------------------------------------


@pytest.fixture
def logged_pyfunc(model_root, tmp_path):
    """Log the tiny model through the real MLflow path, into a temp store."""
    import mlflow

    from msa.train.mlflow_model import (
        BOOSTER_ARTIFACT,
        METADATA_ARTIFACT,
        TimelineModel,
    )

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment("test-pyfunc")
    d = model_root / SEGMENT
    with mlflow.start_run() as run:
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=TimelineModel(),
            artifacts={
                BOOSTER_ARTIFACT: str(d / "model.json"),
                METADATA_ARTIFACT: str(d / "metadata.json"),
            },
        )
    assert run.info.run_id
    return mlflow.pyfunc.load_model(info.model_uri)


def _serving_frame(model_root) -> pd.DataFrame:
    """A frame shaped exactly like what MLflow hands a served model.

    Built by round-tripping through JSON, so the categorical dtype really is
    gone -- which is the whole point of the test.
    """
    from msa.serve.app import request_frame
    from msa.serve.schemas import PermitRequest

    bundle = ModelBundle(SEGMENT, root=model_root)
    X = request_frame(
        [
            PermitRequest(
                jurisdiction="Testville",
                start_date="2024-05-01",
                state="IL",
                system_size_kw=8.4,
            ),
            PermitRequest(jurisdiction="Sampleburg", start_date="2024-06-01"),
        ],
        bundle,
    )
    safe = X.copy()
    for c in CATEGORICAL_FEATURES:
        safe[c] = safe[c].astype("object").where(safe[c].notna(), None)
    round_tripped = pd.read_json(
        json.dumps(safe.to_dict(orient="split"), default=str), orient="split"
    )
    assert not any(
        isinstance(round_tripped[c].dtype, pd.CategoricalDtype)
        for c in round_tripped.columns
    ), "the round trip did not actually drop the categorical dtype"
    return round_tripped[list(FEATURES)]


def test_logged_model_predicts_from_a_json_round_tripped_frame(
    logged_pyfunc, model_root
):
    """The regression: this raised `dtypes ... must be int, float, bool or category`."""
    preds = logged_pyfunc.predict(_serving_frame(model_root))
    preds = np.asarray(preds, dtype=float)
    assert preds.shape == (2,)
    assert np.all(np.isfinite(preds))
    assert np.all(preds >= 0)


def test_logged_model_agrees_with_the_service(logged_pyfunc, model_root):
    """A registry prediction and an API prediction must be the same number."""
    from msa.features.build import back_transform
    from msa.serve.app import request_frame
    from msa.serve.schemas import PermitRequest

    bundle = ModelBundle(SEGMENT, root=model_root)
    items = [
        PermitRequest(
            jurisdiction="Testville",
            start_date="2024-05-01",
            state="IL",
            system_size_kw=8.4,
        ),
        PermitRequest(jurisdiction="Sampleburg", start_date="2024-06-01"),
    ]
    X = request_frame(items, bundle)
    served = back_transform(bundle.model.predict(X), bundle.log_target, bundle.horizon)
    registry = np.asarray(logged_pyfunc.predict(_serving_frame(model_root)), dtype=float)
    np.testing.assert_allclose(registry, served, rtol=1e-6)


def test_logged_model_rejects_a_frame_missing_columns(logged_pyfunc, model_root):
    frame = _serving_frame(model_root).drop(columns=[FEATURES[0]])
    with pytest.raises(Exception, match="missing required columns"):
        logged_pyfunc.predict(frame)
