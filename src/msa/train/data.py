"""Shared data-loading / splitting helpers for every training entrypoint."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from msa.config import PROCESSED_DIR, RANDOM_SEED
from msa.features.build import CATEGORICAL_FEATURES, build_features

LOG = logging.getLogger(__name__)

DATASET_PATH = PROCESSED_DIR / "installations.parquet"

#: Fraction of the (chronologically ordered) data held out as the final test
#: set. The split is by application date, never random, so the reported test
#: metrics answer the question a deployed service actually faces: "how well
#: does a model fitted on the past predict permits filed later?"
TEST_FRACTION = 0.20


@dataclass(frozen=True)
class Dataset:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    meta_train: pd.DataFrame
    meta_test: pd.DataFrame
    split_date: pd.Timestamp
    categories: dict[str, list[str]]

    @property
    def n_train(self) -> int:
        return len(self.X_train)

    @property
    def n_test(self) -> int:
        return len(self.X_test)


def load_raw(path=DATASET_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `python -m msa.ingest.build_dataset` first"
        )
    return pd.read_parquet(path)


def load_tts():
    """Load the LBNL utility-record context layer, or ``None`` if unavailable."""
    try:
        from msa.ingest import tts

        return tts.load()
    except Exception as exc:  # noqa: BLE001
        LOG.warning("Tracking the Sun context unavailable (%s); continuing without", exc)
        return None


def load_dataset(
    path=DATASET_PATH, test_fraction: float = TEST_FRACTION, with_tts: bool = True
) -> Dataset:
    df = load_raw(path)
    tts_df = load_tts() if with_tts else None
    X, y = build_features(df, tts_df)
    meta = df.sort_values(["start_date", "record_id"]).reset_index(drop=True)[
        ["record_id", "source", "jurisdiction", "state", "start_date", "end_date"]
    ]

    cutoff_idx = int(len(X) * (1.0 - test_fraction))
    split_date = meta.loc[cutoff_idx, "start_date"]
    # push the boundary to a clean date so identical days do not straddle folds
    is_test = meta["start_date"] >= split_date

    categories = {
        c: list(map(str, X[c].cat.categories)) for c in CATEGORICAL_FEATURES
    }

    ds = Dataset(
        X_train=X.loc[~is_test].reset_index(drop=True),
        y_train=y.loc[~is_test].reset_index(drop=True),
        X_test=X.loc[is_test].reset_index(drop=True),
        y_test=y.loc[is_test].reset_index(drop=True),
        meta_train=meta.loc[~is_test].reset_index(drop=True),
        meta_test=meta.loc[is_test].reset_index(drop=True),
        split_date=pd.Timestamp(split_date),
        categories=categories,
    )
    LOG.info(
        "dataset: %d train (< %s) / %d test (>= %s)",
        ds.n_train,
        ds.split_date.date(),
        ds.n_test,
        ds.split_date.date(),
    )
    return ds


__all__ = ["Dataset", "load_dataset", "load_raw", "load_tts", "RANDOM_SEED"]
