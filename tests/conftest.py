"""Shared synthetic fixtures. No test touches the network or real data."""
import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.data.panels import Panels


def make_panels(n_days: int = 420, n_tickers: int = 40, seed: int = 0) -> Panels:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    market = pd.Series(rng.normal(0.0003, 0.01, n_days), index=dates)
    beta = rng.uniform(0.5, 1.5, n_tickers)
    ret = market.to_numpy()[:, None] * beta + rng.normal(0, 0.02, (n_days, n_tickers))
    close = pd.DataFrame(50 * np.exp(np.cumsum(ret, axis=0)), index=dates, columns=tickers)
    volume = pd.DataFrame(rng.lognormal(13, 0.5, (n_days, n_tickers)), index=dates, columns=tickers)
    member = pd.DataFrame(True, index=dates, columns=tickers)
    member.iloc[: n_days // 2, :5] = False            # five names join half way
    member.iloc[n_days // 2:, 5:10] = False           # five names leave half way
    sectors = pd.Series([f"Sec{i % 4}" for i in range(n_tickers)], index=tickers)
    return Panels(close, close.copy(), volume, close * volume, member, member.copy(), market, sectors)


@pytest.fixture
def panels() -> Panels:
    return make_panels()


@pytest.fixture(scope="session")
def cfg() -> dict:
    return load_config()
