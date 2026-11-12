"""Amihud (2002) illiquidity."""
import numpy as np
import pandas as pd

from src.data.panels import Panels


def amihud(p: Panels, window: int = 20) -> pd.DataFrame:
    """Log of the `window`-day mean of |return| / dollar volume. Higher = less liquid."""
    ratio = p.returns.abs() / p.dollar_volume.where(p.dollar_volume > 0)
    return np.log(ratio.rolling(window, min_periods=window).mean().where(lambda x: x > 0))
