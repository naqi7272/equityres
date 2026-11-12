"""Stage 4: single-signal evaluation on the research set. Every backtest is logged as a trial."""
import _bootstrap  # noqa: F401
import pandas as pd

from src.backtest.engine import net_returns
from src.config import load_config, parse_args, reports_dir
from src.data.panels import load_panels
from src.evaluation.ic import (daily_rank_ic, ic_by_sector, ic_summary, monotonicity, quantile_returns,
                               signal_autocorrelation, signal_correlation_matrix)
from src.pipeline import backtest_scores, evaluation_start, summary_row
from src.report.io import save_json, save_table
from src.report.plots import plot_correlation, plot_ic_bars, plot_ic_decay
from src.signals.store import load_signals
from src.timing import make_label
from src.trials import log_trial


def main() -> None:
    cfg = load_config(parse_args(__doc__).config)
    panels = load_panels(cfg)
    h, cost = cfg["label"]["horizon"], cfg["costs"]["base_bps"]
    base, variants = cfg["signals"]["base"], cfg["signals"]["variants"]
    signals = load_signals(list(base) + list(variants), cfg)
    start = evaluation_start({n: signals[n] for n in base}, cfg["processing"]["min_names"])
    signals = {n: s.loc[start:] for n, s in signals.items()}
    horizons = sorted(set(cfg["label"]["ic_decay_horizons"]) | {h} | set(cfg["label"]["robustness_horizons"]))
    labels = {k: make_label(panels.adj_close, panels.tradable, k).loc[start:] for k in horizons}

    rows, decay, by_year, by_sector, quantiles = {}, {}, {}, {}, {}
    for name in base:
        sig = signals[name]
        ic = daily_rank_ic(sig, labels[h])
        q = quantile_returns(sig, labels[h], 5)
        rows[name] = {**ic_summary(ic, nw_lags=2 * h), "monotonicity": monotonicity(q),
                      "q5_minus_q1_bps": (q[5] - q[1]) * 1e4,
                      "autocorr_1d": signal_autocorrelation(sig, 1), f"autocorr_{h}d": signal_autocorrelation(sig, h)}
        for k in cfg["label"]["robustness_horizons"]:
            stats = ic_summary(daily_rank_ic(sig, labels[k]), nw_lags=2 * k)
            rows[name].update({f"mean_ic_h{k}": stats["mean_ic"], f"nw_tstat_h{k}": stats["nw_tstat"]})
        decay[name] = {k: daily_rank_ic(sig, labels[k]).mean() for k in cfg["label"]["ic_decay_horizons"]}
        by_year[name] = ic.groupby(ic.index.year).mean()
        by_sector[name] = ic_by_sector(sig, labels[h], panels.sectors)
        quantiles[name] = q * 1e4

    backtests = {}
    for name, spec in {**base, **variants}.items():
        daily = backtest_scores(signals[name], panels, h, cfg)
        trial = {"kind": "single_signal", "signal": name, "func": spec["func"], "params": spec.get("params", {}),
                 "h": h, "cost_bps": cost, "portfolio": cfg["portfolio"], "processing": cfg["processing"]}
        log_trial(cfg, trial, net_returns(daily, cost))
        backtests[name] = {"is_variant": name in variants, **summary_row(daily, panels, cost)}

    summary = pd.DataFrame(rows).T
    summary["net_sharpe"] = [backtests[n]["net_sharpe"] for n in summary.index]
    summary["daily_turnover"] = [backtests[n]["daily_turnover"] for n in summary.index]
    corr = signal_correlation_matrix({n: signals[n] for n in base})
    save_table(summary, "signal_summary", cfg)
    save_table(pd.DataFrame(decay).T, "ic_decay", cfg)
    save_table(pd.DataFrame(by_year), "ic_by_year", cfg)
    save_table(pd.DataFrame(by_sector), "ic_by_sector", cfg)
    save_table(pd.DataFrame(quantiles).T, "quantile_returns_bps", cfg)
    save_table(corr, "signal_correlation", cfg)
    save_table(pd.DataFrame(backtests).T, "single_signal_backtests", cfg)
    save_json({"evaluation_start": str(start.date()), "evaluation_end": str(panels.dates[-1].date()),
               "horizon": h, "nw_lags": 2 * h}, "signal_eval_meta", cfg)

    figs = reports_dir(cfg) / "figures"
    plot_ic_bars(summary, h, figs / "fig2_ic_by_signal.png")
    plot_ic_decay(pd.DataFrame(decay).T, figs / "fig3_ic_decay.png")
    plot_correlation(corr, figs / "appendix_signal_correlation.png")
    print(summary[["mean_ic", "icir", "nw_tstat", "monotonicity", "net_sharpe", "daily_turnover"]].round(3).to_string())


if __name__ == "__main__":
    main()
