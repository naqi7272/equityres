"""Combinatorial purged cross-validation (CPCV), written from scratch.

The dates are cut into N contiguous groups. Every choice of k groups is used
once as the test set, with the rest (purged and embargoed) as training. This
gives C(N, k) splits. Each group is tested in C(N-1, k-1) different splits, so
the test predictions can be recombined into C(N-1, k-1) full-length backtest
paths. The result is a distribution of out-of-sample Sharpe ratios instead of
a single number.
"""
from __future__ import annotations

from itertools import combinations
from math import comb
from typing import Iterator

import numpy as np

from src.validation.purged_kfold import purged_train_index


class CombinatorialPurgedCV:
    def __init__(self, n_groups: int, n_test_groups: int, h: int, embargo: int = 0):
        if not 0 < n_test_groups < n_groups:
            raise ValueError("need 0 < n_test_groups < n_groups")
        self.n_groups, self.k, self.h, self.embargo = n_groups, n_test_groups, h, embargo

    @property
    def n_splits(self) -> int:
        return comb(self.n_groups, self.k)

    @property
    def n_paths(self) -> int:
        return comb(self.n_groups - 1, self.k - 1)

    def groups(self, n_dates: int) -> list[np.ndarray]:
        return np.array_split(np.arange(n_dates), self.n_groups)

    def split(self, n_dates: int) -> Iterator[tuple[np.ndarray, np.ndarray, tuple[int, ...]]]:
        """Yield (train positions, test positions, ids of the test groups)."""
        groups = self.groups(n_dates)
        for combo in combinations(range(self.n_groups), self.k):
            test_idx = np.concatenate([groups[g] for g in combo])
            yield purged_train_index(n_dates, test_idx, self.h, self.embargo), test_idx, combo

    def paths(self) -> list[list[int]]:
        """paths[p][g] = index of the split that supplies group g on path p.

        Path p uses, for each group, the p-th split in which that group is tested.
        """
        seen = [0] * self.n_groups
        table = [[-1] * self.n_groups for _ in range(self.n_paths)]
        for split_id, combo in enumerate(combinations(range(self.n_groups), self.k)):
            for g in combo:
                table[seen[g]][g] = split_id
                seen[g] += 1
        return table
