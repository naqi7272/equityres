"""Helpers shared by the stage scripts."""
from __future__ import annotations

import pandas as pd

from src.backtest.engine import net_returns, run_backtest
from src.backtest.metrics import break_even_cost_bps, performance_summary
from src.data.panels import Panels


def evaluation_start(signals: dict[str, pd.DataFrame], min_names: int) -> pd.Timestamp:
    """First date on which every signal exists for at least `min_names` stocks.

    All evaluation starts here, so no model is scored on a period where some
    signals are still warming up.
    """
    firsts = []
    for frame in signals.values():
        ok = frame.notna().sum(axis=1) >= min_names
        firsts.append(ok.idxmax())
    return max(firsts)


def backtest_scores(scores: pd.DataFrame, panels: Panels, h: int, cfg: dict) -> pd.DataFrame:
    """Backtest a score panel under the project conventions. Returns the daily frame."""
    daily, _ = run_backtest(scores, panels.returns, panels.member, h, cfg["portfolio"]["max_weight"])
    return daily


def summary_row(daily: pd.DataFrame, panels: Panels, cost_bps: float) -> dict:
    """Gross and net headline statistics in one flat dict."""
    gross = performance_summary(daily["gross"], daily["turnover"], panels.market_ret)
    net = performance_summary(net_returns(daily, cost_bps), daily["turnover"], panels.market_ret)
    row = {f"gross_{k}": v for k, v in gross.items() if k in ("ann_return", "sharpe")}
    row.update({f"net_{k}": v for k, v in net.items() if k not in ("daily_turnover", "n_days")})
    row.update({"daily_turnover": net["daily_turnover"], "break_even_bps": break_even_cost_bps(daily),
                "n_days": net["n_days"]})
    return row
