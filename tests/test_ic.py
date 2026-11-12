"""Null test, planted-signal test and Newey-West check."""
import numpy as np
import pandas as pd

from src.evaluation.ic import daily_rank_ic, ic_summary, newey_west_tstat


def _frames(n_days=600, n_names=300, ic=0.0, seed=1):
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=(n_days, n_names))
    label = ic * signal + np.sqrt(1 - ic ** 2) * rng.normal(size=(n_days, n_names))
    idx = pd.bdate_range("2015-01-01", periods=n_days)
    return pd.DataFrame(signal, index=idx), pd.DataFrame(label, index=idx)


def test_null_shuffled_labels_have_zero_ic():
    signal, label = _frames(ic=0.05)
    rng = np.random.default_rng(7)
    shuffled = label.apply(lambda row: pd.Series(rng.permutation(row.to_numpy())), axis=1)
    shuffled.index = label.index
    stats = ic_summary(daily_rank_ic(signal, shuffled), nw_lags=5)
    assert abs(stats["nw_tstat"]) < 3.0
    assert abs(stats["mean_ic"]) < 0.01


def test_planted_ic_is_recovered():
    planted = 0.05
    signal, label = _frames(ic=planted)
    stats = ic_summary(daily_rank_ic(signal, label), nw_lags=5)
    expected_rank_ic = 6 / np.pi * np.arcsin(planted / 2)      # Spearman for a bivariate normal
    assert abs(stats["mean_ic"] - expected_rank_ic) < 0.005
    assert stats["nw_tstat"] > 5


def test_newey_west_equals_plain_tstat_with_zero_lags():
    x = np.random.default_rng(3).normal(0.1, 1, 500)
    plain = x.mean() / (x.std(ddof=0) / np.sqrt(len(x)))
    assert abs(newey_west_tstat(x, lags=0) - plain) < 1e-10


def test_newey_west_shrinks_tstat_for_autocorrelated_series():
    rng = np.random.default_rng(4)
    x = pd.Series(rng.normal(0.2, 1, 2000)).rolling(5).mean().dropna().to_numpy()   # overlapping, like 5-day ICs
    assert abs(newey_west_tstat(x, lags=10)) < abs(newey_west_tstat(x, lags=0)) * 0.7
