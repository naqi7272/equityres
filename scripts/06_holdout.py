"""Stage 6: the single holdout run with the frozen configuration, then the final writeup.

This script refuses to run twice. --force exists only to recover from a crash;
every access is logged and the count is printed in the writeup.
"""
import _bootstrap  # noqa: F401
import numpy as np
import yaml

from src.backtest.engine import net_returns
from src.backtest.metrics import performance_summary
from src.config import config_hash, load_config, parse_args, reports_dir
from src.data.panels import HOLDOUT_CONFIRM, holdout_access_count, holdout_start, load_panels
from src.evaluation.ic import daily_rank_ic, ic_summary
from src.models.combine import make_model
from src.models.dataset import build_dataset
from src.pipeline import backtest_scores, summary_row
from src.report.io import save_json
from src.report.plots import plot_cumulative
from src.report.writeup import build_writeup
from src.signals import compute_signals
from src.timing import label_span, make_label
from src.validation.sharpe_stats import moments, probabilistic_sharpe_ratio


def main() -> None:
    args = parse_args(__doc__, force="rerun although the holdout was already accessed (logged)")
    cfg = load_config(args.config)
    frozen = yaml.safe_load((reports_dir(cfg) / "frozen_config.yaml").read_text())
    if frozen["base_config_hash"] != config_hash(cfg):
        raise SystemExit("Config changed since the model was frozen. Rerun stage 5 or restore the config.")
    if holdout_access_count(cfg) > 0 and not args.force:
        raise SystemExit("The holdout has already been used. It is meant to be touched once.")

    np.random.seed(cfg["seed"])
    panels = load_panels(cfg, include_holdout=True, confirm=HOLDOUT_CONFIRM)
    h, cost = frozen["h"], frozen["cost_bps"]
    start = holdout_start(panels.dates, cfg["split"]["holdout_months"])
    start_pos = panels.dates.get_loc(start)

    signals = compute_signals(panels, cfg, {n: cfg["signals"]["base"][n] for n in frozen["signals"]})
    label = make_label(panels.adj_close, panels.tradable, h)
    ds = build_dataset(signals, label, panels.tradable)

    # Train on research dates only. The last usable training date is the one whose
    # label window ends before the holdout starts, minus the embargo.
    first_valid = int(np.argmax(np.isfinite(ds.y).sum(axis=1) > 0))
    last_train = start_pos - label_span(h) - cfg["validation"]["embargo_days"] - 1
    train_idx = np.arange(first_valid, last_train + 1)
    test_idx = np.arange(start_pos, len(panels.dates))
    model = make_model(frozen["family"], frozen["params"], cfg).fit(ds, train_idx)

    scores = np.full(ds.universe.shape, np.nan)
    scores[test_idx] = np.where(ds.universe[test_idx], model.predict(ds, test_idx), np.nan)
    score_frame = ds.frame(scores)
    daily = backtest_scores(score_frame, panels, h, cfg)
    net = net_returns(daily, cost)
    sr, n_obs, skew, kurt = moments(net)
    ic = daily_rank_ic(score_frame.iloc[test_idx], label.iloc[test_idx])
    result = {
        "holdout_start": str(start.date()), "holdout_end": str(panels.dates[-1].date()),
        "first_return_date": str(daily.index[0].date()), "last_train_date": str(panels.dates[last_train].date()),
        "family": frozen["family"], "params": frozen["params"], "cost_bps": cost,
        "holdout_accesses": holdout_access_count(cfg),
        "gross": performance_summary(daily["gross"], daily["turnover"], panels.market_ret),
        "net": performance_summary(net, daily["turnover"], panels.market_ret),
        "summary": summary_row(daily, panels, cost),
        "psr_vs_zero": probabilistic_sharpe_ratio(sr, n_obs, skew, kurt, 0.0),
        "ic": ic_summary(ic, nw_lags=2 * h),
    }
    save_json(result, "holdout_results", cfg)
    plot_cumulative({"gross": daily["gross"], f"net of {cost} bps per side": net},
                    "Holdout: cumulative return of the frozen configuration", reports_dir(cfg) / "figures" / "fig6_holdout.png")
    build_writeup(cfg)
    print(f"holdout {result['holdout_start']} to {result['holdout_end']}: gross Sharpe {result['gross']['sharpe']:.2f}, "
          f"net Sharpe {result['net']['sharpe']:.2f}, PSR(0) {result['psr_vs_zero']:.3f}, mean IC {result['ic']['mean_ic']:.4f}")
    if result["net"]["sharpe"] > 2.0:
        print("WARNING: net Sharpe above 2. Treat as a bug and look for leakage before believing it.")


if __name__ == "__main__":
    main()
