"""Synthetic raw data in exactly the same on-disk format as the real download.

Used ONLY to smoke-test the pipeline end to end without network access.
Returns have a market factor, sector factors and two weak planted effects
(short-term reversal and low volatility), so the plumbing has something to find.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.download import raw_dir

SECTORS = ["Tech", "Health", "Financials", "Energy", "Industrials", "Staples", "Utilities", "Materials"]


def generate_synthetic_raw(data_dir: Path, start: str, n_tickers: int, benchmark: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    raw = raw_dir(data_dir)
    dates = pd.bdate_range(start, "2025-12-31")
    n_days = len(dates)
    tickers = [f"S{i:03d}" for i in range(n_tickers)]
    sector_id = rng.integers(0, len(SECTORS), n_tickers)

    market = rng.normal(0.0003, 0.01, n_days)
    sector_ret = rng.normal(0, 0.005, (n_days, len(SECTORS)))
    beta = rng.uniform(0.5, 1.5, n_tickers)
    idio_vol = rng.uniform(0.008, 0.03, n_tickers)
    eps = rng.normal(0, 1, (n_days, n_tickers)) * idio_vol
    vol_tilt = -0.01 * (idio_vol - idio_vol.mean())          # low vol earns a little more
    ret = np.zeros((n_days, n_tickers))
    for t in range(n_days):
        past5 = ret[max(0, t - 5):t].sum(axis=0)
        ret[t] = beta * market[t] + sector_ret[t, sector_id] + eps[t] - 0.02 * past5 + vol_tilt
    price = rng.uniform(20, 200, n_tickers) * np.exp(np.cumsum(np.log1p(np.clip(ret, -0.5, 0.5)), axis=0))
    volume = rng.lognormal(14, 0.5, (n_days, n_tickers)) * rng.uniform(0.5, 3, n_tickers)

    # Membership: 70% always in, the rest join or leave once. Some leavers have no price data.
    changes, current, no_data = [], [], {}
    for j, ticker in enumerate(tickers):
        u = rng.uniform()
        lo, hi = 0, n_days
        if u > 0.85:                                         # joins later
            lo = int(rng.integers(300, n_days - 300))
            changes.append({"date": dates[lo], "ticker": ticker, "action": "added"})
            current.append(ticker)
        elif u > 0.70:                                       # leaves, stops trading soon after
            hi = int(rng.integers(300, n_days - 300))
            changes.append({"date": dates[hi], "ticker": ticker, "action": "removed"})
            if rng.uniform() < 0.4:
                no_data[ticker] = "synthetic delisted ticker with no data"
                continue
            hi = min(n_days, hi + 60)
        else:
            current.append(ticker)
        first = max(0, lo - 400)
        frame = pd.DataFrame({
            "open": price[first:hi, j], "high": price[first:hi, j] * 1.01, "low": price[first:hi, j] * 0.99,
            "close": price[first:hi, j], "adj_close": price[first:hi, j], "volume": volume[first:hi, j],
        }, index=dates[first:hi])
        frame.index.name = "date"
        frame.to_parquet(raw / "prices" / f"{ticker}.parquet")

    bench = pd.DataFrame({"adj_close": 100 * np.exp(np.cumsum(np.log1p(market)))}, index=dates)
    for col in ("open", "high", "low", "close"):
        bench[col] = bench["adj_close"]
    bench["volume"] = 1e8
    bench.index.name = "date"
    bench.to_parquet(raw / "prices" / f"{benchmark}.parquet")

    pd.DataFrame({"ticker": current, "security": current,
                  "sector": [SECTORS[sector_id[tickers.index(t)]] for t in current]}
                 ).to_parquet(raw / "universe" / "constituents.parquet")
    pd.DataFrame(changes).to_parquet(raw / "universe" / "changes.parquet")
    (raw / "download_log.json").write_text(json.dumps(no_data, indent=2))
    (raw / "meta.json").write_text(json.dumps({"end": str(dates[-1].date()), "created": "synthetic"}))
