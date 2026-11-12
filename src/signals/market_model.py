"""Rolling market model r_i = a + b * r_m + e, in closed form on wide panels."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_market_model(
    ret: pd.DataFrame, market_ret: pd.Series, window: int, min_periods: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (beta, residual variance) of a trailing-window OLS on the market.

    Row t uses returns in (t-window, t] only, so both outputs are known at close t.
    """
    min_periods = min_periods or window
    mkt = pd.DataFrame(np.where(ret.notna(), market_ret.reindex(ret.index).to_numpy()[:, None], np.nan),
                       index=ret.index, columns=ret.columns)
    roll = lambda x: x.rolling(window, min_periods=min_periods).mean()  # noqa: E731
    mean_r, mean_m = roll(ret), roll(mkt)
    cov = roll(ret * mkt) - mean_r * mean_m
    var_m = roll(mkt * mkt) - mean_m ** 2
    var_r = roll(ret * ret) - mean_r ** 2
    beta = cov / var_m
    resid_var = (var_r - beta ** 2 * var_m).clip(lower=0.0)
    return beta, resid_var
