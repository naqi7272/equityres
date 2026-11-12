"""Stage 1: universe tables from Wikipedia and daily prices from yfinance, cached to parquet."""
import _bootstrap  # noqa: F401
import json

import pandas as pd

from src.config import data_dir, load_config, parse_args
from src.data.download import download_prices, frozen_end_date, raw_dir
from src.data.synthetic import generate_synthetic_raw
from src.data.universe import build_intervals, load_universe_from_wikipedia


def main() -> None:
    cfg = load_config(parse_args(__doc__).config)
    d, ddir = cfg["data"], data_dir(cfg)
    raw = raw_dir(ddir)
    if d.get("synthetic"):
        generate_synthetic_raw(ddir, d["start"], d["synthetic_tickers"], d["benchmark"], cfg["seed"])
        print(f"SYNTHETIC data written to {raw}. These are not market data.")
        return

    constituents, changes, info = load_universe_from_wikipedia(raw / "universe", d["ticker_aliases"], d["max_retries"],
                                                               d.get("membership_source", "auto"))
    constituents.to_parquet(raw / "universe" / "constituents.parquet")
    changes.to_parquet(raw / "universe" / "changes.parquet")
    (raw / "universe" / "source.json").write_text(json.dumps(info, indent=2))
    print(f"change log: {info}")
    intervals, _ = build_intervals(constituents["ticker"].tolist(), changes, pd.Timestamp(d["start"]))
    tickers = sorted(set(intervals["ticker"])) + [d["benchmark"]]
    print(f"{len(constituents)} current constituents, {len(tickers) - 1} historical members since {d['start']}")

    end = frozen_end_date(ddir, d["end"])
    no_data = download_prices(tickers, d["start"], end, ddir, d["batch_size"], d["max_retries"], d["retry_sleep_seconds"])
    have = len(list((raw / "prices").glob("*.parquet")))
    print(f"cache: {have} tickers with data, {len(no_data)} with no data (see {raw / 'download_log.json'})")
    if d["benchmark"] in no_data:
        raise SystemExit("Benchmark download failed. Delete download_log.json and rerun.")


if __name__ == "__main__":
    main()
