"""Stage 3: compute and process all signals on the RESEARCH set only."""
import _bootstrap  # noqa: F401

from src.config import load_config, parse_args
from src.data.panels import load_panels
from src.signals import compute_signals
from src.signals.store import save_signals


def main() -> None:
    cfg = load_config(parse_args(__doc__).config)
    panels = load_panels(cfg)                       # research set; the holdout is cut off in the loader
    specs = {**cfg["signals"]["base"], **cfg["signals"]["variants"]}
    signals = compute_signals(panels, cfg, specs)
    save_signals(signals, cfg)
    print(f"research set: {panels.dates[0].date()} to {panels.dates[-1].date()}, {len(panels.dates)} days")
    for name, frame in signals.items():
        per_day = frame.notna().sum(axis=1)
        print(f"{name:20s} first valid {per_day[per_day > 0].index[0].date()}  avg names/day {per_day[per_day > 0].mean():.0f}")


if __name__ == "__main__":
    main()
