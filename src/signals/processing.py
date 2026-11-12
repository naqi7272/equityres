"""Cross-sectional processing, applied to every signal on every date.

Steps, each using only the names tradable on that date:
  1. winsorize at the 1st and 99th percentile
  2. convert to ranks
  3. map ranks to normal z-scores
  4. neutralize: regress on sector dummies and market beta, keep the residual
  5. rescale the residual to unit standard deviation

Note: winsorizing before ranking only creates ties in the tails. It is kept
because the spec asks for it and it is harmless.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata


def rank_to_z(x: np.ndarray, lower: float = 0.01, upper: float = 0.99) -> np.ndarray:
    """Winsorize, rank and map to normal scores. x must have no NaN."""
    lo, hi = np.quantile(x, [lower, upper])
    ranks = rankdata(np.clip(x, lo, hi), method="average")
    return norm.ppf((ranks - 0.5) / len(x))


def neutralize(z: np.ndarray, exposures: np.ndarray) -> np.ndarray:
    """Residual of an OLS of z on the exposure matrix."""
    coef, *_ = np.linalg.lstsq(exposures, z, rcond=None)
    return z - exposures @ coef


def process_signal(
    raw: pd.DataFrame, tradable: pd.DataFrame, sectors: pd.Series, beta: pd.DataFrame,
    winsor: tuple[float, float] = (0.01, 0.99), do_neutralize: bool = True, min_names: int = 20,
) -> pd.DataFrame:
    """Return the processed z-score panel (NaN outside the tradable universe)."""
    values = raw.to_numpy(dtype=float)
    ok = tradable.reindex_like(raw).to_numpy(dtype=bool) & np.isfinite(values)
    betas = beta.reindex_like(raw).to_numpy(dtype=float)
    dummies = pd.get_dummies(sectors.reindex(raw.columns).fillna("Unknown")).to_numpy(dtype=float)
    out = np.full(values.shape, np.nan)
    for t in range(values.shape[0]):
        idx = np.flatnonzero(ok[t])
        if len(idx) < min_names:
            continue
        z = rank_to_z(values[t, idx], *winsor)
        if do_neutralize:
            d = dummies[idx]
            d = d[:, d.sum(axis=0) > 0]                    # the sector dummies span the intercept
            b = betas[t, idx]
            b = np.where(np.isfinite(b), b, np.nanmean(b) if np.isfinite(b).any() else 0.0)
            b = b - b.mean()
            z = neutralize(z, np.column_stack([d, b]))
            sd = z.std()
            if sd < 1e-12:
                continue
            z = z / sd
        out[t, idx] = z
    return pd.DataFrame(out, index=raw.index, columns=raw.columns)
