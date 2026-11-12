"""Stage 5: combination models under purged CV, costs, CPCV, PSR/DSR/PBO, and the frozen config."""
import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import yaml

from src.backtest.engine import net_returns
from src.backtest.metrics import annualized_sharpe, performance_summary
from src.config import config_hash, load_config, parse_args, reports_dir
from src.data.panels import load_panels
from src.evaluation.ic import newey_west_tstat
from src.models.combine import make_model
from src.models.cv import cpcv_scores, kfold_scores
from src.models.dataset import build_dataset
from src.pipeline import backtest_scores, evaluation_start, summary_row
from src.report.io import save_json, save_table
from src.report.plots import plot_cpcv, plot_cumulative
from src.report.writeup import build_writeup
from src.signals.store import load_signals
from src.timing import make_label
from src.trials import load_trials, log_trial
from src.validation.cpcv import CombinatorialPurgedCV
from src.validation.pbo import probability_of_backtest_overfitting
from src.validation.purged_kfold import PurgedKFold
from src.validation.sharpe_stats import deflated_sharpe_ratio, effective_n_trials, moments, probabilistic_sharpe_ratio

FAMILIES = ["equal_weight", "ic_weighted", "ridge", "lightgbm"]


def main() -> None:
    cfg = load_config(parse_args(__doc__).config)
    np.random.seed(cfg["seed"])
    panels = load_panels(cfg)
    h, cost, val = cfg["label"]["horizon"], cfg["costs"]["base_bps"], cfg["validation"]
    names = list(cfg["signals"]["base"])
    signals = load_signals(names, cfg)
    start = evaluation_start(signals, cfg["processing"]["min_names"])
    signals = {n: s.loc[start:] for n, s in signals.items()}
    tradable = panels.tradable.loc[start:]

    def dataset(horizon: int):
        return build_dataset(signals, make_label(panels.adj_close, panels.tradable, horizon).loc[start:], tradable)

    def trial_config(family: str, params: dict, horizon: int) -> dict:
        return {"kind": "combination", "family": family, "params": params, "signals": names, "h": horizon,
                "cost_bps": cost, "cv": {"n_splits": val["n_splits"], "embargo": val["embargo_days"]},
                "fixed": cfg["models"]["lightgbm_fixed"] if family == "lightgbm" else {},
                "portfolio": cfg["portfolio"], "processing": cfg["processing"]}

    # ---- 1. every model and grid point under the same purged K-fold ----
    ds = dataset(h)
    kfold = PurgedKFold(val["n_splits"], h, val["embargo_days"])
    rows, dailies = [], {}
    for family in FAMILIES:
        for params in cfg["models"][family]:
            daily = backtest_scores(ds.frame(kfold_scores(make_model(family, params, cfg), ds, kfold)), panels, h, cfg)
            trial = trial_config(family, params, h)
            key = log_trial(cfg, trial, net_returns(daily, cost))
            dailies[key] = daily
            rows.append({"family": family, "params": str(params), "hash": key, **summary_row(daily, panels, cost)})
            print(f"{family:13s} {str(params):40s} net Sharpe {rows[-1]['net_sharpe']:.2f}")
    cv = pd.DataFrame(rows).set_index("hash")
    save_table(cv, "cv_results", cfg)

    # ---- 2. robustness horizons, equal weight only (also logged as trials) ----
    robust = []
    for k in cfg["label"]["robustness_horizons"]:
        d = backtest_scores(ds.frame(kfold_scores(make_model("equal_weight", {}, cfg), ds, PurgedKFold(val["n_splits"], k, val["embargo_days"]))), panels, k, cfg)
        log_trial(cfg, trial_config("equal_weight", {}, k), net_returns(d, cost))
        robust.append({"h": k, **summary_row(d, panels, cost)})
    save_table(pd.DataFrame(robust).set_index("h"), "robustness_horizons", cfg)

    # ---- 3. selection by the pre-registered parsimony rule ----
    best_per_family = cv.groupby("family")["net_sharpe"].idxmax()
    ew_key = best_per_family["equal_weight"]
    best_key = cv["net_sharpe"].idxmax()
    diff = (net_returns(dailies[best_key], cost) - net_returns(dailies[ew_key], cost)).dropna()
    t_vs_ew = newey_west_tstat(diff, lags=2 * h) if best_key != ew_key else 0.0
    threshold = cfg["selection"]["min_tstat_over_equal_weight"]
    chosen_key = best_key if (best_key != ew_key and t_vs_ew >= threshold) else ew_key
    chosen = cv.loc[chosen_key]
    chosen_params = next(p for p in cfg["models"][chosen["family"]] if str(p) == chosen["params"])
    chosen_net = net_returns(dailies[chosen_key], cost)

    # ---- 4. cost sensitivity and stability by year for the chosen model ----
    costs = pd.DataFrame([{"cost_bps": c, **performance_summary(net_returns(dailies[chosen_key], c),
                          dailies[chosen_key]["turnover"], panels.market_ret)} for c in cfg["costs"]["grid_bps"]]).set_index("cost_bps")
    save_table(costs, "cost_sensitivity", cfg)
    by_year = pd.DataFrame({cv.loc[k, "family"]: net_returns(dailies[k], cost).groupby(lambda d: d.year).apply(annualized_sharpe)
                            for k in best_per_family[FAMILIES]})
    save_table(by_year, "cv_sharpe_by_year", cfg)

    # ---- 5. CPCV for the best configuration of each family ----
    cpcv = CombinatorialPurgedCV(val["cpcv_groups"], val["cpcv_test_groups"], h, val["embargo_days"])
    cpcv_out, split_sr, path_sr = {}, {}, {}
    for family in FAMILIES:
        params = next(p for p in cfg["models"][family] if str(p) == cv.loc[best_per_family[family], "params"])
        splits, paths = cpcv_scores(make_model(family, params, cfg), ds, cpcv)
        split_sr[family] = [annualized_sharpe(net_returns(backtest_scores(ds.frame(s), panels, h, cfg), cost)
                                              .loc[lambda r: r != 0]) for s in splits]
        path_sr[family] = [annualized_sharpe(net_returns(backtest_scores(ds.frame(p), panels, h, cfg), cost)) for p in paths]
        s, p = np.array(split_sr[family]), np.array(path_sr[family])
        cpcv_out[family] = {"split_mean": s.mean(), "split_std": s.std(ddof=1), "split_min": s.min(), "split_max": s.max(),
                            "split_frac_positive": (s > 0).mean(), "path_mean": p.mean(), "path_std": p.std(ddof=1),
                            "path_min": p.min(), "path_max": p.max()}
        print(f"CPCV {family:13s} splits mean {s.mean():.2f} sd {s.std(ddof=1):.2f} | paths {np.round(p, 2)}")
    save_table(pd.DataFrame(cpcv_out).T, "cpcv_summary", cfg)

    # ---- 6. PSR, deflated Sharpe (raw and effective N), PBO ----
    trials, trial_returns = load_trials(cfg)
    sr, n_obs, skew, kurt = moments(chosen_net)
    eff = effective_n_trials(trial_returns)
    stats = {
        "psr_vs_zero": probabilistic_sharpe_ratio(sr, n_obs, skew, kurt, 0.0),
        "dsr_raw_n": deflated_sharpe_ratio(chosen_net, trials["sharpe_daily"]),
        "dsr_effective_n_avg_corr": deflated_sharpe_ratio(chosen_net, trials["sharpe_daily"], eff["n_eff_avg_corr"]),
        "dsr_effective_n_eigen": deflated_sharpe_ratio(chosen_net, trials["sharpe_daily"], eff["n_eff_eigen"]),
        "effective_n": eff,
        "pbo": probability_of_backtest_overfitting(trial_returns.fillna(0.0).loc[start:], val["pbo_blocks"]),
        "n_trials_distinct": int(len(trials)), "n_trial_records": int(trials["n_evaluations"].sum()),
        "best_trial_sharpe_annualized": float(trials["sharpe_annualized"].max()),
    }
    selection = {
        "chosen_family": chosen["family"], "chosen_params": chosen_params, "chosen_hash": chosen_key,
        "best_cv_family": cv.loc[best_key, "family"], "best_cv_params": cv.loc[best_key, "params"],
        "best_cv_net_sharpe": float(cv.loc[best_key, "net_sharpe"]),
        "equal_weight_net_sharpe": float(cv.loc[ew_key, "net_sharpe"]),
        "tstat_best_vs_equal_weight": float(t_vs_ew), "tstat_threshold": threshold,
        "complex_beats_equal_weight": bool(chosen_key != ew_key),
        "evaluation_start": str(start.date()), "evaluation_end": str(panels.dates[-1].date()),
        "cpcv_splits": cpcv.n_splits, "cpcv_paths": cpcv.n_paths,
    }
    save_json({"selection": selection, "stats": stats}, "combination_results", cfg)

    # ---- 7. freeze the single configuration that will see the holdout ----
    frozen = {"family": chosen["family"], "params": chosen_params, "signals": names, "h": h, "cost_bps": cost,
              "base_config_hash": config_hash(cfg), "trial_hash": chosen_key}
    (reports_dir(cfg) / "frozen_config.yaml").write_text(yaml.safe_dump(frozen, sort_keys=False))

    figs = reports_dir(cfg) / "figures"
    plot_cumulative({cv.loc[k, "family"]: net_returns(dailies[k], cost) for k in best_per_family[FAMILIES]},
                    f"Purged {val['n_splits']}-fold CV: cumulative net return after {cost} bps per side", figs / "fig4_cv_cumulative.png")
    plot_cpcv(split_sr, path_sr, figs / "fig5_cpcv_sharpe.png")
    build_writeup(cfg)
    print(f"\nfrozen: {frozen['family']} {frozen['params']}  |  net Sharpe {chosen['net_sharpe']:.2f}"
          f"  PSR {stats['psr_vs_zero']:.3f}  DSR(raw N={stats['n_trials_distinct']}) {stats['dsr_raw_n']['dsr']:.3f}"
          f"  PBO {stats['pbo']['pbo']:.2f}")
    if chosen["net_sharpe"] > 2.0:
        print("WARNING: net Sharpe above 2. Treat as a bug and look for leakage before believing it.")


if __name__ == "__main__":
    main()
