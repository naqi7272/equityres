"""Daily backtest that follows the timing convention in src/timing.py."""
from __future__ import annotations

import pandas as pd

from src.backtest.portfolio import target_weights
from src.timing import ENTRY_LAG, positions_from_weights


def run_backtest(
    scores: pd.DataFrame, returns: pd.DataFrame, member: pd.DataFrame,
    h: int, max_weight: float, lag: int = ENTRY_LAG,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backtest a score panel. Returns (daily frame, positions).

    daily frame columns:
      gross    - return of day d from the position held at the close of d-1
      turnover - traded notional at the close of d (sum of |position change|)
    Rows start at the first day the book is invested.

    Simplifications, stated openly: weights do not drift with prices between
    rebalances, and a held name with a missing return earns zero (so a
    delisting loss inside the holding period would be missed).
    """
    weights = target_weights(scores, max_weight)
    # A name is dropped at the first close at which it is no longer an index
    # member. Membership at day c is public at c, so this uses no future data.
    positions = positions_from_weights(weights, h, lag).where(member.reindex_like(weights).fillna(False), 0.0)
    gross = (positions.shift(1) * returns.reindex_like(positions).fillna(0.0)).sum(axis=1)
    turnover = positions.diff().abs().sum(axis=1)
    turnover.iloc[0] = positions.iloc[0].abs().sum()
    daily = pd.DataFrame({"gross": gross, "turnover": turnover})
    invested = positions.abs().sum(axis=1) > 0
    if invested.any():
        daily = daily.loc[invested.idxmax():]
    return daily, positions


def net_returns(daily: pd.DataFrame, cost_bps: float) -> pd.Series:
    """Net daily return. `cost_bps` is per side: charged on every dollar traded."""
    return daily["gross"] - cost_bps / 1e4 * daily["turnover"]
