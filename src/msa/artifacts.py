"""The contract between what training writes and what serving is willing to load.

A model directory is only usable if *every* one of these holds:

1. ``metadata.json`` exists, is valid JSON, and carries an
   ``artifact_schema_version`` this build of the code understands.
2. Its ``features`` list is byte-identical to ``msa.features.build.FEATURES``.
   A model fitted before a feature was added or reordered will still *load*
   into XGBoost and still return floats; it will simply return floats about
   a different question. That is the failure mode this check exists to stop.
3. Every categorical feature has a level list, so the serving frame can be
   built with exactly the categories the booster was fitted on.
4. ``model.json`` exists and parses as a booster whose feature count matches.

Anything short of that raises an :class:`ArtifactError`. Each error carries a
``public_reason``, a short, path-free string safe to return to an HTTP client,
separately from ``str(exc)``, which names files and belongs only in the log.

Bump ``ARTIFACT_SCHEMA_VERSION`` whenever the meaning of a field in
``metadata.json`` changes, or whenever a serving-side change stops being able
to read models written by the previous version.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

#: Version of the on-disk model-directory layout. Serving refuses anything else.
ARTIFACT_SCHEMA_VERSION = 1

#: Keys ``metadata.json`` must carry for the service to be able to answer at all.
REQUIRED_METADATA_KEYS: tuple[str, ...] = (
    "artifact_schema_version",
    "segment",
    "horizon_days",
    "features",
    "categorical_features",
    "categories",
    "log1p_target",
    "n_train",
    "train_start",
    "train_end",
    "jurisdictions",
)


class ArtifactError(RuntimeError):
    """Base class: this model directory cannot be served."""

    #: Path-free, exception-text-free summary safe to hand to a client.
    public_reason = "model artifact is unusable"


class ArtifactMissingError(ArtifactError):
    public_reason = "no model artifact has been published for this segment"


class ArtifactCorruptError(ArtifactError):
    public_reason = "model artifact is corrupt or truncated"


class ArtifactSchemaMismatchError(ArtifactError):
    public_reason = "model artifact was written by an incompatible schema version"


def model_fingerprint(model_path: Path) -> str:
    """Content hash of the booster file.

    Used as the stable part of ``model_version``. Unlike an mtime it does not
    change when the file is merely copied, and it *does* change when a single
    byte of the booster changes, which is what a version stamp has to promise.
    """
    h = hashlib.sha256()
    with model_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def build_model_version(segment: str, model_path: Path, trained_at: str) -> str:
    """``<segment>-<YYYYMMDD>-<content hash>``: readable and content-bound."""
    day = trained_at[:10].replace("-", "")
    return f"{segment}-{day}-{model_fingerprint(model_path)}"


def read_metadata(path: Path) -> dict[str, Any]:
    """Load ``metadata.json``, distinguishing 'absent' from 'unreadable'."""
    if not path.exists():
        raise ArtifactMissingError(
            f"{path} not found; run `python -m msa.train.final "
            f"--segment {path.parent.name}` first"
        )
    try:
        raw = path.read_text()
    except OSError as exc:
        raise ArtifactCorruptError(f"{path} could not be read: {exc}") from exc
    if not raw.strip():
        raise ArtifactCorruptError(f"{path} is empty")
    try:
        meta = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ArtifactCorruptError(
            f"{path} is not valid JSON (truncated write?): {exc}"
        ) from exc
    if not isinstance(meta, dict):
        raise ArtifactCorruptError(f"{path} does not contain a JSON object")
    return meta


def validate_metadata(
    meta: dict[str, Any],
    *,
    expected_features: tuple[str, ...],
    expected_categoricals: tuple[str, ...],
    path: Path,
) -> None:
    """Raise unless ``meta`` describes a model this code can serve correctly."""
    version = meta.get("artifact_schema_version")
    if version is None:
        raise ArtifactSchemaMismatchError(
            f"{path} carries no artifact_schema_version; it predates the "
            f"versioned artifact contract (expected {ARTIFACT_SCHEMA_VERSION}). "
            f"Refit with `python -m msa.train.final`."
        )
    if not isinstance(version, int) or version != ARTIFACT_SCHEMA_VERSION:
        raise ArtifactSchemaMismatchError(
            f"{path} has artifact_schema_version={version!r}, this build serves "
            f"{ARTIFACT_SCHEMA_VERSION} only. Refit with "
            f"`python -m msa.train.final`."
        )

    missing = [k for k in REQUIRED_METADATA_KEYS if k not in meta]
    if missing:
        raise ArtifactCorruptError(f"{path} is missing required keys: {missing}")

    features = meta["features"]
    if list(features) != list(expected_features):
        only_artifact = [f for f in features if f not in expected_features]
        only_code = [f for f in expected_features if f not in features]
        raise ArtifactSchemaMismatchError(
            f"{path} feature list does not match msa.features.build.FEATURES "
            f"({len(features)} vs {len(expected_features)} columns; "
            f"in artifact only={only_artifact[:5]}, in code only={only_code[:5]}). "
            f"Serving this model would silently answer a different question. "
            f"Refit with `python -m msa.train.final`."
        )

    categories = meta["categories"]
    if not isinstance(categories, dict):
        raise ArtifactCorruptError(f"{path} 'categories' is not an object")
    missing_levels = [c for c in expected_categoricals if c not in categories]
    if missing_levels:
        raise ArtifactSchemaMismatchError(
            f"{path} has no categorical levels for {missing_levels}; the serving "
            f"frame cannot be encoded the way the booster was fitted."
        )
    for name in expected_categoricals:
        if not isinstance(categories[name], list):
            raise ArtifactCorruptError(
                f"{path} categorical levels for {name!r} are not a list"
            )


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "REQUIRED_METADATA_KEYS",
    "ArtifactCorruptError",
    "ArtifactError",
    "ArtifactMissingError",
    "ArtifactSchemaMismatchError",
    "build_model_version",
    "model_fingerprint",
    "read_metadata",
    "validate_metadata",
]
