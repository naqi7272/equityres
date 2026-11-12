"""Low volatility (Blitz and van Vliet 2007)."""
import pandas as pd

from src.data.panels import Panels


def low_vol(p: Panels, window: int = 60) -> pd.DataFrame:
    """Negative of realized daily-return volatility over `window` days."""
    return -p.returns.rolling(window, min_periods=window).std()
