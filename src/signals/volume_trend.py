"""Volume trend (high-volume return premium, Gervais, Kaniel and Mingelgrin 2001)."""
import pandas as pd

from src.data.panels import Panels


def volume_trend(p: Panels, short: int = 20, long: int = 120) -> pd.DataFrame:
    """Short-window average dollar volume over long-window average dollar volume."""
    dv = p.dollar_volume
    return dv.rolling(short, min_periods=short).mean() / dv.rolling(long, min_periods=long).mean()
