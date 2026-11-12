"""Purged K-fold cross-validation with embargo, written from scratch.

Splits are over DATE positions 0..n-1, never over rows. A date stands for the
whole cross-section of that day.

Why purge: the label of date i is the return over closes i+lag .. i+lag+h
(see src/timing.py). Two dates closer than h days share part of that return
path. If one is in train and the other in test, the model has seen part of the
test label. So for a test block [a, b] we drop every training date i with
a-h <= i <= b+h. That removes all dates whose label window touches the test
label windows (including sharing a single price, which is conservative).

Why embargo: features are autocorrelated beyond the label window, so we drop
a further `embargo` dates after each test block.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np


def contiguous_blocks(idx: np.ndarray) -> list[tuple[int, int]]:
    """Split sorted integer positions into inclusive (start, end) runs."""
    idx = np.sort(np.asarray(idx))
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    return list(zip(starts.tolist(), ends.tolist()))


def purged_train_index(n: int, test_idx: np.ndarray, h: int, embargo: int) -> np.ndarray:
    """Training date positions left after purging and embargoing around the test set."""
    keep = np.ones(n, dtype=bool)
    for a, b in contiguous_blocks(test_idx):
        keep[max(0, a - h): min(n, b + h + embargo + 1)] = False
    return np.flatnonzero(keep)


class PurgedKFold:
    """K contiguous test folds over dates, with purged and embargoed training sets."""

    def __init__(self, n_splits: int, h: int, embargo: int = 0):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        self.n_splits, self.h, self.embargo = n_splits, h, embargo

    def split(self, n_dates: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (train positions, test positions)."""
        for test_idx in np.array_split(np.arange(n_dates), self.n_splits):
            yield purged_train_index(n_dates, test_idx, self.h, self.embargo), test_idx
