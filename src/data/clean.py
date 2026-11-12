"""Turn the raw cache into clean wide panels, and log everything that is dropped."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.download import raw_dir
from src.data.panels import Panels
from src.data.universe import build_intervals, membership_matrix


def load_raw_universe(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    uni = raw_dir(data_dir) / "universe"
    return pd.read_parquet(uni / "constituents.parquet"), pd.read_parquet(uni / "changes.parquet")


def _load_wide(data_dir: Path, tickers: list[str], calendar: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    cols: dict[str, dict[str, pd.Series]] = {"close": {}, "adj_close": {}, "volume": {}}
    for ticker in tickers:
        file = raw_dir(data_dir) / "prices" / f"{ticker}.parquet"
        if file.exists():
            frame = pd.read_parquet(file)
            for name in cols:
                cols[name][ticker] = frame[name]
    return {name: pd.DataFrame(series).reindex(calendar) for name, series in cols.items()}


def repair_split_artifacts(
    adj_close: pd.DataFrame, close: pd.DataFrame, max_abs_return: float, tolerance: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Find extreme one-day moves and repair the ones that look like unadjusted splits.

    A move is a suspected split artifact when 1+r is within `tolerance` of k or
    1/k for an integer k in 2..10. We then back-adjust all earlier prices by k,
    which is what a correct split adjustment does. Earlier returns do not
    change, so this is causal. Other extreme moves are kept (they may be real)
    but the name is not traded on that signal date. Every event is logged.
    """
    adj_close, close = adj_close.copy(), close.copy()
    ratios = np.array([k for k in range(2, 11)] + [1 / k for k in range(2, 11)])
    events = []
    for _ in range(5):                                    # a ticker can have several artifacts
        ret = adj_close.pct_change(fill_method=None)
        rows, cols = np.where(ret.abs().to_numpy() > max_abs_return)
        repaired = False
        for i, j in zip(rows, cols):
            gross = 1.0 + ret.iat[i, j]
            nearest = ratios[np.argmin(np.abs(gross / ratios - 1.0))]
            is_split = abs(gross / nearest - 1.0) < tolerance
            key = (ret.index[i], ret.columns[j])
            if is_split:
                adj_close.iloc[:i, j] *= nearest
                close.iloc[:i, j] *= nearest
                repaired = True
            if is_split or key not in {(e["date"], e["ticker"]) for e in events}:
                events.append({"date": key[0], "ticker": key[1], "return": float(gross - 1.0),
                               "action": f"repaired as {nearest:.4g}:1 split" if is_split else "kept, flagged"})
        if not repaired:
            break
    log = pd.DataFrame(events, columns=["date", "ticker", "return", "action"])
    flagged = pd.DataFrame(False, index=adj_close.index, columns=adj_close.columns)
    for e in events:
        flagged.at[e["date"], e["ticker"]] = True
    return adj_close, close, flagged, log


def survivorship_by_year(intervals: pd.DataFrame, member: pd.DataFrame, adj_close: pd.DataFrame,
                         calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Per year: how many historical members have no price data at all, and the
    share of member-days that have a price. This quantifies survivorship bias."""
    with_data = set(adj_close.columns)
    all_member = membership_matrix(intervals, calendar, sorted(intervals["ticker"].unique()))
    rows = []
    for year, block in all_member.groupby(all_member.index.year):
        in_year = block.columns[block.any()]
        missing = [t for t in in_year if t not in with_data]
        member_days = int(block.to_numpy().sum())
        cols = [t for t in in_year if t in with_data]
        priced_days = int((block[cols] & adj_close.loc[block.index, cols].notna()).to_numpy().sum())
        rows.append({
            "year": int(year), "members_in_year": len(in_year), "members_no_data": len(missing),
            "pct_members_no_data": 100.0 * len(missing) / max(len(in_year), 1),
            "avg_members_per_day": member_days / len(block),
            "pct_member_days_priced": 100.0 * priced_days / max(member_days, 1),
        })
    return pd.DataFrame(rows)


def build_panels(cfg: dict, data_dir: Path) -> tuple[Panels, dict, dict[str, pd.DataFrame]]:
    """Build clean panels. Returns (panels, quality summary, detail tables)."""
    q, liq = cfg["quality"], cfg["liquidity"]
    constituents, changes = load_raw_universe(data_dir)
    bench = pd.read_parquet(raw_dir(data_dir) / "prices" / f"{cfg['data']['benchmark']}.parquet")
    calendar = pd.DatetimeIndex(bench.index[bench.index >= cfg["data"]["start"]])
    market_ret = bench["adj_close"].pct_change().reindex(calendar)

    intervals, issues = build_intervals(constituents["ticker"].tolist(), changes, calendar[0])
    wide = _load_wide(data_dir, sorted(intervals["ticker"].unique()), calendar)
    adj_close, close, volume = wide["adj_close"], wide["close"], wide["volume"]
    tickers = list(adj_close.columns)

    bad_price = (adj_close <= 0) | (close <= 0)
    adj_close, close = adj_close.mask(bad_price), close.mask(bad_price)
    adj_close, close, extreme, extreme_log = repair_split_artifacts(
        adj_close, close, q["max_abs_return"], q["split_tolerance"])

    # Missing days: no price on a trading day between a ticker's first and last price.
    has = adj_close.notna()
    inside = has.cumsum().gt(0) & has[::-1].cumsum()[::-1].gt(0)
    missing_inside = inside & ~has
    zero_volume = has & ~(volume > 0)

    member = membership_matrix(intervals, calendar, tickers)
    dollar_volume = close * volume
    # Liquidity uses only data up to and including day t (rolling windows look back).
    median_dv = dollar_volume.rolling(liq["window"], min_periods=liq["window"]).median()
    liquid = (close >= liq["min_price"]) & (median_dv >= liq["min_median_dollar_volume"])
    tradable = member & liquid & has & ~zero_volume & ~extreme

    sectors = constituents.set_index("ticker")["sector"].reindex(tickers).fillna("Unknown")
    panels = Panels(adj_close, close, volume, dollar_volume, member, tradable, market_ret, sectors)

    md = member.to_numpy()
    summary = {
        "first_date": str(calendar[0].date()), "last_date": str(calendar[-1].date()),
        "n_trading_days": len(calendar),
        "n_historical_members": int(intervals["ticker"].nunique()),
        "n_members_with_data": len(tickers),
        "n_members_no_data": int(intervals["ticker"].nunique() - len(tickers)),
        "n_membership_inconsistencies": len(issues),
        "n_nonpositive_prices": int(bad_price.to_numpy().sum()),
        "n_missing_days_inside_history": int(missing_inside.to_numpy().sum()),
        "n_zero_volume_days": int(zero_volume.to_numpy().sum()),
        "n_extreme_returns": int(len(extreme_log)),
        "n_split_repairs": int(extreme_log["action"].str.startswith("repaired").sum()),
        "member_days": int(md.sum()),
        "member_days_dropped_no_price": int((md & ~has.to_numpy()).sum()),
        "member_days_dropped_illiquid": int((md & has.to_numpy() & ~liquid.to_numpy()).sum()),
        "member_days_dropped_zero_volume": int((md & zero_volume.to_numpy()).sum()),
        "member_days_dropped_extreme_return": int((md & extreme.to_numpy()).sum()),
        "member_days_tradable": int(tradable.to_numpy().sum()),
        "avg_tradable_per_day": float(tradable.sum(axis=1).mean()),
        "n_unknown_sector": int((sectors == "Unknown").sum()),
    }
    source_file = raw_dir(data_dir) / "universe" / "source.json"
    source = json.loads(source_file.read_text()) if source_file.exists() else {}
    summary["change_log_source"] = source.get("change_log_source", "live article")
    summary["change_log_snapshot_date"] = source.get("snapshot_date")
    summary["n_membership_events_approximated"] = source.get("n_approximated_events", 0)
    no_data_file = raw_dir(data_dir) / "download_log.json"
    reasons = json.loads(no_data_file.read_text()) if no_data_file.exists() else {}
    missing_tickers = sorted(set(intervals["ticker"]) - set(tickers))
    details = {
        "survivorship_by_year": survivorship_by_year(intervals, member, adj_close, calendar),
        "extreme_returns": extreme_log,
        "membership_inconsistencies": pd.DataFrame(issues, columns=["date", "ticker", "issue"]),
        "members_without_data": pd.DataFrame(
            {"ticker": missing_tickers, "reason": [reasons.get(t, "") for t in missing_tickers]}),
        "intervals": intervals,
    }
    return panels, summary, details
