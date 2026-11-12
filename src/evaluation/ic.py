"""Single-signal diagnostics: rank IC, Newey-West t-stat, decay, quantiles, turnover."""
from __future__ import annotations

import numpy as np
import pandas as pd


def daily_rank_ic(signal: pd.DataFrame, label: pd.DataFrame, min_names: int = 20) -> pd.Series:
    """Spearman correlation between signal and label across stocks, one value per date."""
    valid = signal.notna() & label.notna()
    rs = signal.where(valid).rank(axis=1)
    rl = label.where(valid).rank(axis=1)
    rs = rs.sub(rs.mean(axis=1), axis=0)
    rl = rl.sub(rl.mean(axis=1), axis=0)
    ic = (rs * rl).sum(axis=1) / np.sqrt((rs ** 2).sum(axis=1) * (rl ** 2).sum(axis=1))
    return ic.where(valid.sum(axis=1) >= min_names)


def newey_west_tstat(x: pd.Series | np.ndarray, lags: int) -> float:
    """t-statistic of the mean with Newey-West (Bartlett kernel) standard errors.

    Daily ICs of an h-day label are autocorrelated up to lag h-1 because the
    label windows overlap. A plain t-stat would be too large, so lags >= h.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    d = x - x.mean()
    long_run_var = (d @ d) / n
    for lag in range(1, min(lags, n - 1) + 1):
        long_run_var += 2.0 * (1.0 - lag / (lags + 1.0)) * (d[lag:] @ d[:-lag]) / n
    return float(x.mean() / np.sqrt(long_run_var / n)) if long_run_var > 0 else float("nan")


def ic_summary(ic: pd.Series, nw_lags: int) -> dict:
    ic = ic.dropna()
    sd = ic.std()
    return {"mean_ic": float(ic.mean()), "ic_std": float(sd), "icir": float(ic.mean() / sd) if sd > 0 else 0.0,
            "nw_tstat": newey_west_tstat(ic, nw_lags), "hit_rate": float((ic > 0).mean()), "n_days": int(len(ic))}


def quantile_returns(signal: pd.DataFrame, label: pd.DataFrame, n_quantiles: int = 5) -> pd.Series:
    """Mean demeaned label per signal quantile (1 = lowest signal), averaged over dates."""
    valid = signal.notna() & label.notna()
    pct = signal.where(valid).rank(axis=1, pct=True)
    bucket = np.ceil(pct * n_quantiles).clip(1, n_quantiles)
    out = {}
    for q in range(1, n_quantiles + 1):
        out[q] = label.where(bucket == q).mean(axis=1).mean()
    return pd.Series(out, name="mean_label")


def monotonicity(quantile_means: pd.Series) -> float:
    """Spearman correlation between quantile number and mean return (+1 = perfectly monotone)."""
    return float(quantile_means.rank().corr(pd.Series(quantile_means.index, index=quantile_means.index), method="pearson"))


def signal_autocorrelation(signal: pd.DataFrame, lag: int) -> float:
    """Average cross-sectional rank correlation between the signal at t and t-lag."""
    return float(daily_rank_ic(signal, signal.shift(lag)).mean())


def signal_correlation_matrix(signals: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Average daily cross-sectional correlation between each pair of processed signals."""
    names = list(signals)
    out = pd.DataFrame(1.0, index=names, columns=names)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            out.loc[a, b] = out.loc[b, a] = float(daily_rank_ic(signals[a], signals[b]).mean())
    return out


def ic_by_sector(signal: pd.DataFrame, label: pd.DataFrame, sectors: pd.Series, min_names: int = 8) -> pd.Series:
    out = {}
    for sector, tickers in sectors.groupby(sectors).groups.items():
        cols = [t for t in tickers if t in signal.columns]
        if len(cols) >= min_names:
            out[sector] = daily_rank_ic(signal[cols], label[cols], min_names).mean()
    return pd.Series(out, name="mean_ic")
