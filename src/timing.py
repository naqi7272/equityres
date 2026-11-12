"""THE timing convention. Every label, purge window and position in the project
is derived from the functions in this file, so there is one place to audit.

    * A signal for date t uses data up to and including the close of day t.
    * The position is entered at the close of day t+1 (ENTRY_LAG = 1 full day).
    * The label is the return from close t+1 to close t+1+h.
    * So a signal at t first earns the return of day t+2 (close t+1 -> close t+2).

Why the lag: a signal that needs the close of day t cannot realistically be
traded at that same close. One full day is conservative and easy to defend.
"""
from __future__ import annotations

import pandas as pd

ENTRY_LAG: int = 1


def forward_return(close: pd.DataFrame, h: int, lag: int = ENTRY_LAG) -> pd.DataFrame:
    """Raw label: return from close t+lag to close t+lag+h, indexed by signal date t."""
    return close.shift(-(lag + h)) / close.shift(-lag) - 1.0


def label_span(h: int, lag: int = ENTRY_LAG) -> int:
    """Number of trading days after t that the label of date t depends on."""
    return lag + h


def demean_cross_section(frame: pd.DataFrame, universe: pd.DataFrame | None = None) -> pd.DataFrame:
    """Subtract the same-date cross-sectional mean (over the universe, if given)."""
    if universe is not None:
        frame = frame.where(universe)
    return frame.sub(frame.mean(axis=1), axis=0)


def make_label(close: pd.DataFrame, universe: pd.DataFrame, h: int, lag: int = ENTRY_LAG) -> pd.DataFrame:
    """Cross-sectionally demeaned forward return for names in the universe at t."""
    return demean_cross_section(forward_return(close, h, lag), universe)


def positions_from_weights(weights: pd.DataFrame, h: int, lag: int = ENTRY_LAG) -> pd.DataFrame:
    """Position held at the close of day c (it earns the return of day c+1).

    Overlapping holding periods: the book is the average of the h target
    portfolios formed from signals at c-lag, ..., c-lag-h+1. Each is entered
    `lag` days after its signal and held for h days.
    """
    return weights.fillna(0.0).shift(lag, fill_value=0.0).rolling(h, min_periods=1).sum() / h
