"""Proximity to the 52-week high (George and Hwang 2004)."""
import pandas as pd

from src.data.panels import Panels


def high_52w(p: Panels, window: int = 252) -> pd.DataFrame:
    """Close divided by the highest close of the past `window` days (at most 1)."""
    return p.adj_close / p.adj_close.rolling(window, min_periods=window).max()
