"""Portfolio construction: dollar-neutral long-short weights from a score."""
from __future__ import annotations

import numpy as np
import pandas as pd


def target_weights(scores: pd.DataFrame, max_weight: float) -> pd.DataFrame:
    """Weights proportional to the demeaned score, 50% long and 50% short, capped per name.

    The cap is applied by clipping and rescaling each side a few times. If a
    side has too few names to reach 50% under the cap, gross stays below 1
    rather than breaking the cap.
    """
    s = scores.sub(scores.mean(axis=1), axis=0).fillna(0.0).to_numpy()
    long_side, short_side = np.clip(s, 0, None), np.clip(-s, 0, None)

    def scale(side: np.ndarray) -> np.ndarray:
        for _ in range(6):
            total = side.sum(axis=1, keepdims=True)
            side = np.divide(0.5 * side, total, out=np.zeros_like(side), where=total > 0)
            side = np.minimum(side, max_weight)
        return side

    return pd.DataFrame(scale(long_side) - scale(short_side), index=scores.index, columns=scores.columns)
