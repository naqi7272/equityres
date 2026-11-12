"""Save and load processed signal panels."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import data_dir


def signal_dir(cfg: dict) -> Path:
    path = data_dir(cfg) / "processed" / "signals"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_signals(signals: dict[str, pd.DataFrame], cfg: dict) -> None:
    for name, frame in signals.items():
        frame.astype("float32").to_parquet(signal_dir(cfg) / f"{name}.parquet")


def load_signals(names: list[str], cfg: dict) -> dict[str, pd.DataFrame]:
    return {name: pd.read_parquet(signal_dir(cfg) / f"{name}.parquet").astype("float64") for name in names}
