"""Short-term reversal (Jegadeesh 1990, Lehmann 1990)."""
import pandas as pd

from src.data.panels import Panels


def reversal(p: Panels, window: int = 5) -> pd.DataFrame:
    """Negative of the past `window`-day return."""
    return -(p.adj_close / p.adj_close.shift(window) - 1.0)
