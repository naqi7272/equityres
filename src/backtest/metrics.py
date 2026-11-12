"""Performance statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.validation.sharpe_stats import moments

TRADING_DAYS = 252


def annualized_sharpe(returns: pd.Series) -> float:
    sd = returns.std()
    return float(returns.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0


def max_drawdown(returns: pd.Series) -> float:
    """Largest peak-to-trough loss of the cumulative (additive) return, as a negative number."""
    wealth = returns.cumsum()
    return float((wealth - wealth.cummax()).min())


def performance_summary(returns: pd.Series, turnover: pd.Series, market_ret: pd.Series) -> dict:
    """Headline statistics for one daily return series."""
    r = returns.dropna()
    _, _, skew, kurt = moments(r)
    mkt = market_ret.reindex(r.index)
    ok = mkt.notna()
    beta = float(np.cov(r[ok], mkt[ok])[0, 1] / mkt[ok].var()) if ok.sum() > 2 else float("nan")
    return {
        "ann_return": float(r.mean() * TRADING_DAYS),
        "ann_vol": float(r.std() * np.sqrt(TRADING_DAYS)),
        "sharpe": annualized_sharpe(r),
        "max_drawdown": max_drawdown(r),
        "daily_turnover": float(turnover.reindex(r.index).mean()),
        "hit_rate": float((r > 0).mean()),
        "skew": skew, "kurtosis": kurt, "market_beta": beta, "n_days": int(len(r)),
    }


def break_even_cost_bps(daily: pd.DataFrame) -> float:
    """Per-side cost in bps at which the mean net return is zero."""
    turnover = daily["turnover"].mean()
    return float(daily["gross"].mean() / turnover * 1e4) if turnover > 0 else float("nan")
