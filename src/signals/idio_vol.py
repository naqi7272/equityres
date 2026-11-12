"""Idiosyncratic volatility (Ang, Hodrick, Xing and Zhang 2006)."""
import numpy as np
import pandas as pd

from src.data.panels import Panels
from src.signals.market_model import rolling_market_model


def idio_vol(p: Panels, window: int = 60) -> pd.DataFrame:
    """Negative residual volatility from a market-model fit on the trailing window."""
    _, resid_var = rolling_market_model(p.returns, p.market_ret, window)
    return -np.sqrt(resid_var)
