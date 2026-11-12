"""Probabilistic and Deflated Sharpe Ratio (Bailey and Lopez de Prado), from scratch.

All Sharpe ratios in this file are NON-annualized (per period, here per day).
Kurtosis is raw kurtosis (3 for a normal distribution).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

EULER_GAMMA = 0.5772156649015329


def sharpe_ratio(returns: np.ndarray | pd.Series) -> float:
    """Per-period Sharpe ratio: mean over sample standard deviation."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    return float(r.mean() / sd) if sd > 0 else 0.0


def moments(returns: np.ndarray | pd.Series) -> tuple[float, int, float, float]:
    """(per-period Sharpe, T, skewness, raw kurtosis) of a return series."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    d = r - r.mean()
    m2 = (d ** 2).mean()
    skew = (d ** 3).mean() / m2 ** 1.5
    kurt = (d ** 4).mean() / m2 ** 2
    return sharpe_ratio(r), len(r), float(skew), float(kurt)


def probabilistic_sharpe_ratio(sr: float, n_obs: int, skew: float, kurt: float, sr_benchmark: float = 0.0) -> float:
    """PSR(SR*) = P(true Sharpe > SR*), given the estimate and the shape of the returns.

    PSR = Phi( (SR - SR*) * sqrt(T - 1) / sqrt(1 - skew*SR + (kurt - 1)/4 * SR^2) )
    Negative skew and fat tails widen the error of the Sharpe estimate.
    """
    denom = np.sqrt(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2)
    return float(norm.cdf((sr - sr_benchmark) * np.sqrt(n_obs - 1) / denom))


def expected_max_sharpe(sr_variance: float, n_trials: float) -> float:
    """Expected maximum Sharpe among N trials that all have true Sharpe zero.

    SR* = sqrt(V) * ((1 - g) * PhiInv(1 - 1/N) + g * PhiInv(1 - 1/(N*e)))
    V is the variance of the trial Sharpe ratios, g the Euler-Mascheroni constant.
    """
    if n_trials <= 1:
        return 0.0
    g = EULER_GAMMA
    z = (1 - g) * norm.ppf(1 - 1 / n_trials) + g * norm.ppf(1 - 1 / (n_trials * np.e))
    return float(np.sqrt(sr_variance) * z)


def deflated_sharpe_ratio(returns: np.ndarray | pd.Series, trial_sharpes: np.ndarray, n_trials: float | None = None) -> dict:
    """DSR = PSR evaluated at the expected maximum Sharpe under the null.

    `trial_sharpes` are the per-period Sharpe ratios of every trial that was run.
    `n_trials` defaults to their count; pass an effective N to allow for
    correlated trials.
    """
    trial_sharpes = np.asarray(trial_sharpes, dtype=float)
    n = float(len(trial_sharpes) if n_trials is None else n_trials)
    variance = float(trial_sharpes.var(ddof=1)) if len(trial_sharpes) > 1 else 0.0
    sr_star = expected_max_sharpe(variance, n)
    sr, n_obs, skew, kurt = moments(returns)
    return {"dsr": probabilistic_sharpe_ratio(sr, n_obs, skew, kurt, sr_star), "sr": sr,
            "sr_star": sr_star, "n_trials": n, "trial_sr_variance": variance, "n_obs": n_obs,
            "skew": skew, "kurt": kurt}


def effective_n_trials(trial_returns: pd.DataFrame) -> dict:
    """Effective number of independent trials from the correlation of trial returns.

    Two estimates:
      * average correlation:  N_eff = rho + (1 - rho) * N   (Bailey and Lopez de Prado 2014)
      * eigenvalue participation ratio:  N_eff = (sum l)^2 / sum l^2
    Thirty variants of the same idea are not thirty independent bets. Raw N
    over-penalizes; effective N is the fairer but assumption-laden figure.
    """
    corr = trial_returns.corr().to_numpy()
    n = corr.shape[0]
    if n < 2:
        return {"n_raw": n, "avg_corr": 1.0, "n_eff_avg_corr": float(n), "n_eff_eigen": float(n)}
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    avg = float(corr[np.triu_indices(n, 1)].mean())
    eig = np.clip(np.linalg.eigvalsh(corr), 0, None)
    return {"n_raw": n, "avg_corr": avg,
            "n_eff_avg_corr": float(min(n, max(1.0, avg + (1 - avg) * n))),
            "n_eff_eigen": float(eig.sum() ** 2 / (eig ** 2).sum())}
