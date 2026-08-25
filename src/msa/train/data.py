"""Shared data-loading / splitting helpers for every training entrypoint."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import pandas as pd

from msa.config import DEFAULT_SEGMENT, PROCESSED_DIR, RANDOM_SEED
from msa.features.build import CATEGORICAL_FEATURES, build_features, clean

LOG = logging.getLogger(__name__)

DATASET_PATH = PROCESSED_DIR / "installations.parquet"

#: Fraction of the data held out as the final test set. The split is always by
#: date, never random, so the reported test metrics answer the question a
#: deployed service actually faces: "how well does a model fitted on the past
#: predict projects that start later?"
TEST_FRACTION = 0.20

#: ``global``          one cut-off date across the whole corpus.
#: ``by_jurisdiction`` each jurisdiction contributes its own most-recent 20%.
#:
#: ``by_jurisdiction`` is the default because the corpus is assembled from
#: registers whose coverage windows differ by up to a decade (Saint Paul stops
#: in 2025-06, Overland Park starts in 2024-01). A single global cut-off
#: therefore changes the *mix of cities* between train and test as well as the
#: time period, which confounds "does this generalise forward in time?" with
#: "does this generalise to a different city?". Splitting inside each
#: jurisdiction isolates the temporal question. The global split is still
#: computed and reported alongside it; see reports/final_metrics.json.
DEFAULT_SPLIT_STRATEGY = "by_jurisdiction"


def _holdout_mask(
    meta: pd.DataFrame, test_fraction: float, strategy: str
) -> tuple[pd.Series, pd.Timestamp]:
    if strategy == "global":
        cutoff_idx = int(len(meta) * (1.0 - test_fraction))
        split_date = meta["start_date"].sort_values().iloc[cutoff_idx]
        return meta["start_date"] >= split_date, pd.Timestamp(split_date)

    if strategy != "by_jurisdiction":
        raise ValueError(f"unknown split strategy {strategy!r}")

    is_test = pd.Series(False, index=meta.index)
    for _, idx in meta.groupby("jurisdiction", sort=False).groups.items():
        sub = meta.loc[idx].sort_values("start_date")
        k = int(len(sub) * (1.0 - test_fraction))
        if k >= len(sub):
            continue
        cutoff = sub["start_date"].iloc[k]
        is_test.loc[idx] = meta.loc[idx, "start_date"] >= cutoff
    median_cut = meta.loc[is_test, "start_date"].min()
    return is_test, pd.Timestamp(median_cut)


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
    segment: str

    @property
    def n_train(self) -> int:
        return len(self.X_train)

    @property
    def n_test(self) -> int:
        return len(self.X_test)


def load_raw(path=DATASET_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run `python -m msa.ingest.build_dataset` first"
        )
    return pd.read_parquet(path)


def load_tts():
    """Load the LBNL utility-record context layer, or ``None`` if unavailable.

    ``MSA_DISABLE_TTS=1`` skips it outright. The layer is a ~2 GB public
    download the first time; CI and the smoke pipeline need a run that touches
    no network at all, and "no market-context features" is a supported
    configuration (the columns simply stay missing, which XGBoost handles).
    """
    if os.environ.get("MSA_DISABLE_TTS", "").lower() in {"1", "true", "yes"}:
        LOG.info("MSA_DISABLE_TTS set; building features without the TTS layer")
        return None
    try:
        from msa.ingest import tts

        return tts.load()
    except Exception as exc:
        LOG.warning("Tracking the Sun context unavailable (%s); continuing without", exc)
        return None


def load_dataset(
    path=DATASET_PATH,
    test_fraction: float = TEST_FRACTION,
    with_tts: bool = True,
    segment: str = DEFAULT_SEGMENT,
    split_strategy: str = DEFAULT_SPLIT_STRATEGY,
) -> Dataset:
    df = clean(load_raw(path), segment=segment)
    tts_df = load_tts() if with_tts else None
    X, y = build_features(df, tts_df)
    meta = df.sort_values(["start_date", "record_id"]).reset_index(drop=True)[
        ["record_id", "source", "jurisdiction", "state", "start_date", "end_date"]
    ]

    is_test, split_date = _holdout_mask(meta, test_fraction, split_strategy)

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
        segment=segment,
    )
    LOG.info(
        "dataset: %d train (< %s) / %d test (>= %s)",
        ds.n_train,
        ds.split_date.date(),
        ds.n_test,
        ds.split_date.date(),
    )
    return ds


__all__ = ["RANDOM_SEED", "Dataset", "load_dataset", "load_raw", "load_tts"]
