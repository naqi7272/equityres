"""Config loading. config/base.yaml is the single source of truth; an optional
override file (e.g. config/synthetic.yaml) is deep-merged on top of it."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def load_config(override: str | Path | None = None) -> dict:
    """Load config/base.yaml, optionally merged with an override yaml."""
    cfg = yaml.safe_load((ROOT / "config" / "base.yaml").read_text())
    if override:
        cfg = _deep_merge(cfg, yaml.safe_load(Path(override).read_text()) or {})
    return cfg


def data_dir(cfg: dict) -> Path:
    return ROOT / cfg["paths"]["data_dir"]


def reports_dir(cfg: dict) -> Path:
    path = ROOT / cfg["paths"]["reports_dir"]
    (path / "tables").mkdir(parents=True, exist_ok=True)
    (path / "figures").mkdir(parents=True, exist_ok=True)
    return path


def config_hash(obj: Any) -> str:
    """Stable short hash of any json-serialisable object."""
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def parse_args(description: str, **extra_flags: str) -> argparse.Namespace:
    """Common CLI: --config plus optional boolean flags {name: help}."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=None, help="override yaml, e.g. config/synthetic.yaml")
    for name, help_text in extra_flags.items():
        parser.add_argument(f"--{name.replace('_', '-')}", action="store_true", help=help_text)
    return parser.parse_args()
