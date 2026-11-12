"""Timing test, universe test, cost and staggering checks."""
import numpy as np
import pandas as pd

from src.backtest.engine import net_returns, run_backtest
from src.backtest.metrics import annualized_sharpe, break_even_cost_bps
from src.backtest.portfolio import target_weights
from src.evaluation.ic import daily_rank_ic, ic_summary
from src.timing import forward_return, make_label, positions_from_weights


def _iid_world(n_days=1500, n_names=100, seed=5):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2012-01-01", periods=n_days)
    ret = pd.DataFrame(rng.normal(0, 0.02, (n_days, n_names)), index=idx)
    close = (1 + ret).cumprod()
    member = pd.DataFrame(True, index=idx, columns=ret.columns)
    return ret, close, member


def test_timing_knowing_tomorrows_return_gives_zero_edge():
    """A 'signal' at t that is literally the return of day t+1. Under the
    convention the position is entered at close t+1, after that return is
    over, so the edge must be zero. With no lag the same signal is a money
    machine, which proves the test can detect a timing bug."""
    ret, close, member = _iid_world()
    cheat = ret.shift(-1)

    ic = ic_summary(daily_rank_ic(cheat, make_label(close, member, h=1)), nw_lags=5)
    assert abs(ic["nw_tstat"]) < 3.0
    daily, _ = run_backtest(cheat, ret, member, h=1, max_weight=0.05)
    assert abs(annualized_sharpe(daily["gross"])) < 1.0

    leaky, _ = run_backtest(cheat, ret, member, h=1, max_weight=0.05, lag=0)
    assert annualized_sharpe(leaky["gross"]) > 10.0


def test_label_matches_convention():
    _, close, _ = _iid_world(n_days=50, n_names=3)
    label = forward_return(close, h=5)
    t = 10
    expected = close.iloc[t + 6] / close.iloc[t + 1] - 1
    assert np.allclose(label.iloc[t], expected)
    assert label.iloc[-6:].isna().all().all()


def test_universe_no_position_outside_membership(panels):
    rng = np.random.default_rng(0)
    scores = pd.DataFrame(rng.normal(size=panels.adj_close.shape), index=panels.dates,
                          columns=panels.adj_close.columns).where(panels.tradable)
    _, positions = run_backtest(scores, panels.returns, panels.member, h=5, max_weight=0.1)
    assert (positions.abs().sum(axis=1) > 0).sum() > 100
    assert positions.where(~panels.member, 0.0).abs().to_numpy().max() == 0.0


def test_weights_are_dollar_neutral_unit_gross_and_capped():
    rng = np.random.default_rng(1)
    scores = pd.DataFrame(rng.standard_t(3, size=(20, 300)))
    scores.iloc[:, :50] = np.nan
    w = target_weights(scores, max_weight=0.01)
    assert np.allclose(w.sum(axis=1), 0.0, atol=1e-3)
    assert np.allclose(w.abs().sum(axis=1), 1.0, atol=1e-3)
    assert w.abs().to_numpy().max() <= 0.01 + 1e-12
    assert (w.iloc[:, :50] == 0).all().all()


def test_staggered_book_averages_h_lagged_portfolios():
    idx = pd.bdate_range("2020-01-01", periods=12)
    w = pd.DataFrame({"A": np.arange(12.0)}, index=idx)
    pos = positions_from_weights(w, h=3)
    assert np.isclose(pos["A"].iloc[5], (4 + 3 + 2) / 3)      # signals of days 4, 3, 2; never day 5
    assert pos["A"].iloc[0] == 0.0


def test_costs_and_break_even():
    daily = pd.DataFrame({"gross": [0.001, 0.001], "turnover": [0.5, 0.5]})
    assert np.allclose(net_returns(daily, 5), 0.001 - 0.0005 * 0.5)
    assert np.isclose(break_even_cost_bps(daily), 20.0)
    assert np.allclose(net_returns(daily, 20), 0.0)
