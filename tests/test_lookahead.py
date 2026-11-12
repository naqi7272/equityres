"""Lookahead test: corrupt all data after date t, recompute, and require that
every signal value up to t is bit-for-bit unchanged."""
import numpy as np
import pandas as pd

from src.signals import compute_signals
from tests.conftest import make_panels


def _corrupt_after(panels, cut: int, seed: int = 99):
    rng = np.random.default_rng(seed)
    out = make_panels()
    for name in ("adj_close", "close", "volume", "dollar_volume"):
        frame = getattr(out, name)
        frame.iloc[cut + 1:] = frame.iloc[cut + 1:].to_numpy() * rng.uniform(0.2, 5.0, frame.iloc[cut + 1:].shape)
    out.market_ret.iloc[cut + 1:] = rng.normal(0, 0.05, len(out.market_ret) - cut - 1)
    for name in ("member", "tradable"):
        frame = getattr(out, name)
        frame.iloc[cut + 1:] = rng.uniform(size=frame.iloc[cut + 1:].shape) > 0.5
    return out


def test_no_signal_uses_future_data(panels, cfg):
    specs = {**cfg["signals"]["base"], **cfg["signals"]["variants"]}
    cut = 380
    clean = compute_signals(panels, cfg, specs)
    dirty = compute_signals(_corrupt_after(panels, cut), cfg, specs)
    for name in specs:
        a, b = clean[name].iloc[: cut + 1], dirty[name].iloc[: cut + 1]
        assert a.notna().to_numpy().sum() > 0, f"{name} is empty, the test would be vacuous"
        pd.testing.assert_frame_equal(a, b, check_exact=True, obj=name)
        # The corruption must actually change later values, or the test has no power.
        assert not np.allclose(clean[name].iloc[cut + 1:].fillna(0), dirty[name].iloc[cut + 1:].fillna(0)), name
