"""Probability of backtest overfitting via combinatorially symmetric
cross-validation (Bailey, Borwein, Lopez de Prado and Zhu 2017), from scratch."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def _sharpe_cols(x: np.ndarray) -> np.ndarray:
    sd = x.std(axis=0, ddof=1)
    return np.divide(x.mean(axis=0), sd, out=np.zeros(x.shape[1]), where=sd > 0)


def probability_of_backtest_overfitting(trial_returns: pd.DataFrame, n_blocks: int = 10) -> dict:
    """PBO = share of splits where the in-sample best trial is below the OOS median.

    The T x N matrix of trial returns is cut into `n_blocks` time blocks. For
    every way of choosing half of the blocks as in-sample, we pick the trial
    with the best in-sample Sharpe and record its relative rank w out of
    sample. The logit log(w / (1 - w)) is negative when it falls below the
    median.
    """
    x = trial_returns.dropna().to_numpy()
    n_trials = x.shape[1]
    blocks = np.array_split(np.arange(x.shape[0]), n_blocks)
    logits = []
    for combo in combinations(range(n_blocks), n_blocks // 2):
        in_idx = np.concatenate([blocks[b] for b in combo])
        out_idx = np.concatenate([blocks[b] for b in range(n_blocks) if b not in combo])
        best = int(np.argmax(_sharpe_cols(x[in_idx])))
        w = rankdata(_sharpe_cols(x[out_idx]))[best] / (n_trials + 1)
        logits.append(np.log(w / (1 - w)))
    logits = np.array(logits)
    return {"pbo": float((logits <= 0).mean()), "n_splits": len(logits), "n_trials": n_trials,
            "median_logit": float(np.median(logits))}
