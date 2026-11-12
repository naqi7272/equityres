"""Signal registry. Every signal is a pure function of a Panels object: the
value on row t depends only on rows <= t (enforced by tests/test_lookahead.py)."""
from __future__ import annotations

import pandas as pd

from src.data.panels import Panels
from src.signals.amihud import amihud
from src.signals.high_52w import high_52w
from src.signals.idio_vol import idio_vol
from src.signals.low_vol import low_vol
from src.signals.market_model import rolling_market_model
from src.signals.momentum import momentum
from src.signals.processing import process_signal
from src.signals.residual_momentum import residual_momentum
from src.signals.reversal import reversal
from src.signals.volume_trend import volume_trend

FUNCTIONS = {
    "momentum": momentum, "reversal": reversal, "low_vol": low_vol, "idio_vol": idio_vol,
    "residual_momentum": residual_momentum, "high_52w": high_52w, "amihud": amihud,
    "volume_trend": volume_trend,
}


def market_beta(panels: Panels, cfg: dict) -> pd.DataFrame:
    s = cfg["signals"]
    beta, _ = rolling_market_model(panels.returns, panels.market_ret, s["beta_window"], s["beta_min_periods"])
    return beta


def compute_signals(panels: Panels, cfg: dict, specs: dict[str, dict]) -> dict[str, pd.DataFrame]:
    """Compute and process the signals in `specs` ({name: {func, params}})."""
    proc = cfg["processing"]
    beta = market_beta(panels, cfg)
    out = {}
    for name, spec in specs.items():
        raw = FUNCTIONS[spec["func"]](panels, **spec.get("params", {}))
        out[name] = process_signal(
            raw, panels.tradable, panels.sectors, beta,
            winsor=(proc["winsor_lower"], proc["winsor_upper"]),
            do_neutralize=proc["neutralize"], min_names=proc["min_names"])
    return out
