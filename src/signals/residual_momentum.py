"""Residual momentum (Blitz, Huij and Martens 2011), market-model version."""
import pandas as pd

from src.data.panels import Panels
from src.signals.market_model import rolling_market_model


def residual_momentum(p: Panels, lookback: int = 252, skip: int = 21, beta_window: int = 252) -> pd.DataFrame:
    """Sum of market-model residual returns over t-lookback..t-skip, scaled by their volatility.

    The residual on day s uses the beta estimated up to day s-1, so each
    residual is out of sample with respect to its own beta.
    """
    ret = p.returns
    beta, _ = rolling_market_model(ret, p.market_ret, beta_window, min_periods=beta_window // 2)
    resid = ret - beta.shift(1).mul(p.market_ret.reindex(ret.index), axis=0)
    n = lookback - skip
    total = resid.rolling(n, min_periods=n).sum().shift(skip)
    vol = resid.rolling(n, min_periods=n).std().shift(skip)
    return total / vol
