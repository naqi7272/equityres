"""The Panels container and the research / holdout loader.

All panels are wide DataFrames: index = trading dates, columns = tickers.
"""
from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import data_dir, reports_dir

HOLDOUT_CONFIRM = "I_AM_RUNNING_THE_SINGLE_HOLDOUT_EVALUATION"
_FRAMES = ("adj_close", "close", "volume", "dollar_volume", "member", "tradable")


@dataclass
class Panels:
    adj_close: pd.DataFrame      # split and dividend adjusted close (for returns)
    close: pd.DataFrame          # split adjusted close (for the price filter)
    volume: pd.DataFrame
    dollar_volume: pd.DataFrame  # close * volume
    member: pd.DataFrame         # bool: in the S&P 500 on that date
    tradable: pd.DataFrame       # bool: member & liquid & clean data
    market_ret: pd.Series        # daily return of the market proxy
    sectors: pd.Series           # ticker -> GICS sector (current snapshot)

    @property
    def returns(self) -> pd.DataFrame:
        """Daily simple returns of the adjusted close."""
        return self.adj_close.pct_change(fill_method=None)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.adj_close.index

    def truncate(self, last_date: pd.Timestamp) -> "Panels":
        """Copy with every time series cut at last_date (inclusive)."""
        kwargs = {}
        for f in fields(self):
            val = getattr(self, f.name)
            kwargs[f.name] = val if f.name == "sectors" else val.loc[:last_date]
        return Panels(**kwargs)


def panel_dir(cfg: dict) -> Path:
    path = data_dir(cfg) / "processed" / "panels"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_panels(panels: Panels, cfg: dict) -> None:
    out = panel_dir(cfg)
    for name in _FRAMES:
        getattr(panels, name).to_parquet(out / f"{name}.parquet")
    panels.market_ret.to_frame("market_ret").to_parquet(out / "market_ret.parquet")
    panels.sectors.to_frame("sector").to_parquet(out / "sectors.parquet")


def _read_all(cfg: dict) -> Panels:
    src = panel_dir(cfg)
    frames = {name: pd.read_parquet(src / f"{name}.parquet") for name in _FRAMES}
    market = pd.read_parquet(src / "market_ret.parquet")["market_ret"]
    sectors = pd.read_parquet(src / "sectors.parquet")["sector"]
    return Panels(market_ret=market, sectors=sectors, **frames)


def holdout_start(dates: pd.DatetimeIndex, holdout_months: int) -> pd.Timestamp:
    """First trading date of the final holdout (the last `holdout_months` months)."""
    cutoff = dates[-1] - pd.DateOffset(months=holdout_months)
    return dates[dates > cutoff][0]


def holdout_log_path(cfg: dict) -> Path:
    return reports_dir(cfg) / "holdout_access.log"


def load_panels(cfg: dict, include_holdout: bool = False, confirm: str | None = None) -> Panels:
    """Load panels. By default ONLY the research set is returned.

    The holdout is physically cut off here, so research code cannot touch it
    by accident. To get it you must pass include_holdout=True AND the exact
    confirmation string, and every such access is appended to a log file.
    """
    panels = _read_all(cfg)
    start = holdout_start(panels.dates, cfg["split"]["holdout_months"])
    if not include_holdout:
        return panels.truncate(panels.dates[panels.dates < start][-1])
    if confirm != HOLDOUT_CONFIRM:
        raise PermissionError("Holdout access requires confirm=HOLDOUT_CONFIRM. It is meant to happen once.")
    caller = inspect.stack()[1]
    record = {
        "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "caller": f"{Path(caller.filename).name}:{caller.lineno}",
        "holdout_start": str(start.date()),
        "holdout_end": str(panels.dates[-1].date()),
    }
    with holdout_log_path(cfg).open("a") as fh:
        fh.write(json.dumps(record) + "\n")
    return panels


def holdout_access_count(cfg: dict) -> int:
    path = holdout_log_path(cfg)
    return len(path.read_text().splitlines()) if path.exists() else 0
