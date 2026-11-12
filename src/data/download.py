"""Download daily OHLCV from yfinance into a per-ticker parquet cache.

The cache is the contract: one file per ticker with columns
open, high, low, close, adj_close, volume. A ticker is never downloaded twice.
Tickers that return nothing (mostly delisted names) are recorded in
download_log.json with the reason, which feeds the survivorship report.
"""
from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

import pandas as pd

COLUMNS = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Adj Close": "adj_close", "Volume": "volume"}


def raw_dir(data_dir: Path) -> Path:
    path = data_dir / "raw"
    (path / "prices").mkdir(parents=True, exist_ok=True)
    (path / "universe").mkdir(parents=True, exist_ok=True)
    return path


def frozen_end_date(data_dir: Path, configured_end: str | None) -> str:
    """The end date is fixed on the first download so later reruns see the same data."""
    meta_file = raw_dir(data_dir) / "meta.json"
    if meta_file.exists():
        return json.loads(meta_file.read_text())["end"]
    end = configured_end or date.today().isoformat()
    meta_file.write_text(json.dumps({"end": end, "created": date.today().isoformat()}, indent=2))
    return end


def _split_batch(df: pd.DataFrame, batch: list[str]) -> dict[str, pd.DataFrame]:
    """Turn the yfinance multi-ticker frame into {ticker: clean frame}."""
    out: dict[str, pd.DataFrame] = {}
    if df is None or df.empty:
        return out
    for ticker in batch:
        try:
            sub = df[ticker] if isinstance(df.columns, pd.MultiIndex) else df
        except KeyError:
            continue
        sub = sub.rename(columns=COLUMNS)
        sub = sub[[c for c in COLUMNS.values() if c in sub.columns]].dropna(how="all")
        if len(sub) and "adj_close" in sub.columns:
            sub.index = pd.to_datetime(sub.index).tz_localize(None)
            sub.index.name = "date"
            out[ticker] = sub.astype("float64")
    return out


def download_prices(
    tickers: list[str], start: str, end: str, data_dir: Path,
    batch_size: int = 40, max_retries: int = 4, retry_sleep: float = 5.0,
) -> dict[str, str]:
    """Fill the cache for `tickers`. Returns {ticker: reason} for tickers with no data."""
    import yfinance as yf

    raw = raw_dir(data_dir)
    log_file = raw / "download_log.json"
    no_data: dict[str, str] = json.loads(log_file.read_text()) if log_file.exists() else {}
    todo = [t for t in tickers if not (raw / "prices" / f"{t}.parquet").exists() and t not in no_data]
    print(f"{len(tickers)} tickers requested, {len(todo)} to download")

    for attempt in range(max_retries):
        if not todo:
            break
        # After the first pass, retry the failures in small batches with a longer pause:
        # an empty answer can be a rate limit rather than a missing ticker.
        size = batch_size if attempt == 0 else max(1, batch_size // 8)
        failed: list[str] = []
        for i in range(0, len(todo), size):
            batch = todo[i:i + size]
            try:
                df = yf.download(batch, start=start, end=end, auto_adjust=False, actions=False,
                                 group_by="ticker", threads=True, progress=False)
            except Exception as exc:  # noqa: BLE001
                print(f"  batch failed ({exc!r}); will retry")
                df = None
            got = _split_batch(df, batch)
            for ticker, frame in got.items():
                frame.to_parquet(raw / "prices" / f"{ticker}.parquet")
            failed += [t for t in batch if t not in got]
            print(f"  pass {attempt + 1}: {min(i + size, len(todo))}/{len(todo)} done, {len(failed)} empty")
            time.sleep(1.0 if attempt == 0 else retry_sleep)
        todo = failed
        if todo and attempt < max_retries - 1:
            time.sleep(retry_sleep * (2 ** attempt))

    errors = getattr(getattr(yf, "shared", None), "_ERRORS", {}) or {}
    for ticker in todo:
        no_data[ticker] = str(errors.get(ticker, "no data returned after retries"))[:200]
    log_file.write_text(json.dumps(no_data, indent=2, sort_keys=True))
    return no_data
