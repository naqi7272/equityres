"""Momentum 12-1 (Jegadeesh and Titman 1993)."""
import pandas as pd

from src.data.panels import Panels


def momentum(p: Panels, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """Return from t-lookback to t-skip. The last month is skipped to avoid the reversal effect."""
    return p.adj_close.shift(skip) / p.adj_close.shift(lookback) - 1.0
