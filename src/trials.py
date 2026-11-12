"""Append-only trial registry.

Every strategy configuration evaluated on research data is recorded here:
single signals, parameter variants and combination models. The count of
distinct configurations is the N of the deflated Sharpe ratio, and the stored
daily returns give the correlation used for the effective N.

Records are never deleted. Re-running a stage appends again; the count uses
distinct config hashes, so a rerun of the same configuration is not double
counted, while any new configuration always is.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import ROOT, config_hash, data_dir
from src.validation.sharpe_stats import sharpe_ratio


def trials_file(cfg: dict) -> Path:
    path = ROOT / cfg["paths"]["trials_file"]
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _returns_dir(cfg: dict) -> Path:
    path = data_dir(cfg) / "trials"
    path.mkdir(parents=True, exist_ok=True)
    return path


def log_trial(cfg: dict, trial_config: dict, net_returns: pd.Series) -> str:
    """Append one trial and store its daily net returns. Returns the config hash."""
    h = config_hash(trial_config)
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_hash": h, "config": trial_config,
        "sharpe_daily": sharpe_ratio(net_returns),
        "sharpe_annualized": sharpe_ratio(net_returns) * 252 ** 0.5,
        "n_days": int(net_returns.notna().sum()),
    }
    with trials_file(cfg).open("a") as fh:
        fh.write(json.dumps(record, default=str) + "\n")
    net_returns.rename(h).to_frame().to_parquet(_returns_dir(cfg) / f"{h}.parquet")
    return h


def load_trials(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (latest record per distinct config, T x N matrix of trial returns)."""
    path = trials_file(cfg)
    if not path.exists():
        return pd.DataFrame(), pd.DataFrame()
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    table = pd.DataFrame(records).drop_duplicates("config_hash", keep="last").reset_index(drop=True)
    table["n_evaluations"] = pd.DataFrame(records).groupby("config_hash").size().reindex(table["config_hash"]).to_numpy()
    series = [pd.read_parquet(_returns_dir(cfg) / f"{h}.parquet")[h] for h in table["config_hash"]]
    return table, pd.concat(series, axis=1)
