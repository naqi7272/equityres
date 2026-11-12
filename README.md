# Cross-sectional equity signal research with leak-free validation

**Question.** Do simple price and volume signals predict the cross-section of
forward returns in US large-cap equities after costs, and does a combination of
them survive purged cross-validation and a multiple-testing correction?

The project is judged on method, not on the size of the Sharpe ratio. A modest,
honest number with airtight validation is the goal.

## Headline result

<!-- RESULTS:START -->
Frozen configuration: **equal_weight** . Costs: 5 bps per side. Distinct trials logged: 29.

| Metric | Research set (purged CV) | Holdout (single run) |
|---|---|---|
| Sharpe, gross | 0.11 | 0.11 |
| Sharpe, net | -0.09 | -0.04 |
| Ann. return, net | -0.4% | -0.3% |
| Ann. vol | 4.7% | 6.4% |
| Max drawdown | -18.4% | -8.6% |
| Daily turnover | 7.5% | 7.4% |
| Break-even cost (bps/side) | 2.8 | 3.6 |
| Market beta | -0.04 | 0.02 |
| Days | 3361 | 500 |

PSR vs 0 (research): 0.369 | DSR raw N: 0.001 | DSR effective N: 0.002 | PBO: 0.06 | PSR vs 0 (holdout): 0.478
<!-- RESULTS:END -->

Full writeup: [`reports/writeup.md`](reports/writeup.md) (generated). Data
report: `reports/data_report.md`.

## Timing convention (defined once, in `src/timing.py`)

| Step | When |
|---|---|
| Signal for day t | uses data up to and including the close of t |
| Entry | close of t+1 (one full day lag) |
| Label | return from close t+1 to close t+1+h, demeaned across stocks |
| Horizon | h = 5 trading days; h = 1 and h = 21 as robustness checks |

## Method in brief

- **Data.** Daily OHLCV from yfinance since 2010, cached to parquet. S&P 500
  membership is rebuilt point-in-time by walking the index change history back
  from today's Wikipedia constituents. Wikipedia removed its change log from
  the live article, so the history comes from the public dataset
  `fja05680/sp500` (dated member lists, itself compiled from Wikipedia); an old
  Wikipedia revision can be used instead via `data.membership_source`. A stock is traded only while it is a member and passes a price and
  dollar-volume filter that uses past data only.
- **Signals (8).** Momentum 12-1, 5-day reversal, low volatility,
  idiosyncratic volatility, residual momentum, 52-week-high proximity, Amihud
  illiquidity, volume trend. Per date: winsorize, rank, z-score, neutralize
  against sector dummies and market beta. Signs are fixed in advance.
- **Validation, all written from scratch** (`src/validation`): purged K-fold
  with embargo, combinatorial purged CV, probabilistic and deflated Sharpe
  ratio, probability of backtest overfitting.
- **Trial registry.** Every configuration evaluated on research data is
  appended to `trials.jsonl`. Its count and Sharpe variance feed the deflated
  Sharpe ratio.
- **Combination.** Equal weight, IC-weighted, ridge, LightGBM, all under the
  same purged CV. A fitted model replaces equal weight only if it beats it with
  t >= 2 (rule fixed in `config/base.yaml` before any result).
- **Backtest.** Dollar-neutral, unit gross, 1% name cap, h staggered
  sub-portfolios, 5 bps per side with a 0 to 20 bps sensitivity table.
- **Holdout.** The last 24 months. The loader cuts them off unless an explicit
  confirmation string is passed, and logs each access. Stage 6 refuses to run
  twice.

## How to run (Windows, Git Bash, Python 3.12)

```bash
python -m venv .venv
source .venv/Scripts/activate        # Linux or macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q                  # all tests must pass before any result is read

python scripts/01_download.py        # Wikipedia + yfinance, cached (slowest step)
python scripts/02_build_data.py      # panels, cleaning, data report
python scripts/03_signals.py         # signals on the research set only
python scripts/04_evaluate_signals.py
python scripts/05_combine_backtest.py   # purged CV, CPCV, DSR, PBO, frozen config
python scripts/06_holdout.py         # ONCE, at the very end
```

`bash scripts/run_all.sh` runs the tests and stages 1 to 5.

Offline smoke test on synthetic data (no network, about 2 minutes). Output goes
to git-ignored `data_synthetic/` and `reports_synthetic/` and is never a result:

```bash
bash scripts/run_all.sh config/synthetic.yaml
```

All parameters live in `config/base.yaml`. Seeds are fixed and package versions
are pinned.

## Tests

| Test | What it proves |
|---|---|
| `test_lookahead.py` | Corrupting all data after t leaves every signal value up to t unchanged |
| `test_purged_cv.py` | No training label window overlaps a test fold or its embargo (K-fold and CPCV) |
| `test_backtest.py` (timing) | A signal that knows tomorrow's return earns nothing under the lag; with lag 0 it earns a Sharpe above 10 |
| `test_backtest.py` (universe) | No position is held outside membership dates |
| `test_ic.py` | Shuffled labels give zero IC; a planted IC is recovered |
| `test_sharpe_stats.py` | PSR, expected maximum Sharpe and DSR match hand-computed values |
| `test_data.py` | Membership walk-back, Wikipedia parsing, split repair, holdout lock and access log |
| `test_models.py` | Models read training dates only; CV covers every date once |

## Design choices to defend

- **One-day entry lag.** A close-of-day signal cannot be traded at that same close.
- **Purge width.** Labels span closes t+1..t+1+h, so training dates within h
  days of a test block share return days with it and are removed.
- **Embargo.** Features are autocorrelated beyond the label window.
- **Normal-score target for ridge and LightGBM.** A per-date transform, so no
  information crosses dates, and outliers do not dominate the fit.
- **Raw N versus effective N in the DSR.** Raw N treats 29 correlated variants
  as independent and over-penalizes. Effective N corrects for correlation but
  depends on the estimator, so both are reported.
- **Trial counting.** Distinct configuration hashes. A rerun is not counted
  twice, a new variant always is.
- **Split repair.** A one-day move within 2% of a k:1 ratio is treated as an
  unadjusted split and back-adjusted, which leaves earlier returns unchanged.

## Known limitations

- **Survivorship bias.** Delisted tickers are mostly missing on yfinance. The
  data report counts them by year. This is the main weakness.
- Sectors are a current snapshot, not point-in-time.
- The price filter uses split-adjusted prices, which embed later splits.
- Recycled tickers can map to a different company on Yahoo.
- Flat cost per side, no market impact, no borrow cost, no weight drift
  between rebalances.

## Layout

```
config/          yaml configs (base.yaml is the single source of truth)
src/timing.py    the timing convention
src/data/        download, cache, universe, cleaning, research/holdout loader
src/signals/     one module per signal, plus processing
src/evaluation/  IC, Newey-West, quantiles, turnover
src/validation/  purged k-fold, CPCV, PSR, DSR, PBO
src/models/      combination methods and CV runner
src/backtest/    portfolio construction, costs, metrics
src/report/      plots, tables, generated writeup
src/trials.py    append-only trial registry
scripts/         01_download ... 06_holdout
tests/
reports/         figures, tables, writeup (generated)
```
