"""Stack processed signals and labels into arrays for the combination models."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.evaluation.ic import daily_rank_ic
from src.signals.processing import rank_to_z


@dataclass
class Dataset:
    dates: pd.DatetimeIndex
    tickers: pd.Index
    names: list[str]
    z: np.ndarray          # (T, N, K) processed signals, missing values set to 0 (neutral)
    universe: np.ndarray   # (T, N) bool, tradable on the signal date
    y: np.ndarray          # (T, N) training target: cross-sectional normal score of the label
    daily_ic: np.ndarray   # (T, K) rank IC of each signal on each date

    def frame(self, scores: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(scores, index=self.dates, columns=self.tickers)


def build_dataset(signals: dict[str, pd.DataFrame], label: pd.DataFrame, tradable: pd.DataFrame) -> Dataset:
    """The target is the label turned into a normal score within each date.

    This is a per-date transform (no information crosses dates) and it stops
    a few extreme returns from dominating the ridge and tree fits.
    """
    names = list(signals)
    universe = tradable.to_numpy(dtype=bool)
    z = np.stack([signals[n].to_numpy(dtype=np.float32) for n in names], axis=-1)
    z = np.where(np.isfinite(z) & universe[..., None], z, 0.0).astype(np.float32)
    lab = label.to_numpy(dtype=float)
    y = np.full(lab.shape, np.nan, dtype=np.float32)
    for t in range(lab.shape[0]):
        idx = np.flatnonzero(universe[t] & np.isfinite(lab[t]))
        if len(idx) >= 20:
            y[t, idx] = rank_to_z(lab[t, idx], 0.0, 1.0)
    ic = np.column_stack([daily_rank_ic(signals[n], label).to_numpy() for n in names])
    return Dataset(label.index, label.columns, names, z, universe, y, ic)
