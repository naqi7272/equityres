"""Small helpers to write and read result tables. The writeup reads only these files."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import reports_dir


def save_table(df: pd.DataFrame, name: str, cfg: dict, index: bool = True) -> Path:
    path = reports_dir(cfg) / "tables" / f"{name}.csv"
    df.to_csv(path, index=index, float_format="%.6g")
    return path


def load_table(name: str, cfg: dict, index_col: int | None = 0) -> pd.DataFrame:
    return pd.read_csv(reports_dir(cfg) / "tables" / f"{name}.csv", index_col=index_col)


def save_json(obj: dict, name: str, cfg: dict) -> Path:
    path = reports_dir(cfg) / "tables" / f"{name}.json"
    path.write_text(json.dumps(obj, indent=2, default=str))
    return path


def load_json(name: str, cfg: dict) -> dict:
    return json.loads((reports_dir(cfg) / "tables" / f"{name}.json").read_text())


def has_result(name: str, cfg: dict) -> bool:
    return (reports_dir(cfg) / "tables" / name).exists()


def to_markdown(df: pd.DataFrame, fmt: dict[str, str] | None = None, default: str = "{:.3f}", index: bool = True) -> str:
    """Render a DataFrame as a markdown table (no extra dependency)."""
    fmt = fmt or {}
    if index:
        df = df.reset_index()
    cells = []
    for col in df.columns:
        f = fmt.get(col, default)
        cells.append([f.format(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and pd.notna(v) else str(v)
                      for v in df[col]])
    lines = ["| " + " | ".join(str(c) for c in df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(row) + " |" for row in zip(*cells)]
    return "\n".join(lines)
