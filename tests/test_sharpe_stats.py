"""PSR and DSR against hand-computed values.

The expected numbers were worked out by hand from the formulas, using normal
table values (Phi and its inverse), not by calling the code under test.
"""
import math
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from src.validation.pbo import probability_of_backtest_overfitting
from src.validation.sharpe_stats import (deflated_sharpe_ratio, effective_n_trials, expected_max_sharpe,
                                         moments, probabilistic_sharpe_ratio)


def test_psr_normal_returns():
    # SR=0.1, T=101, skew=0, kurt=3: z = 0.1*10 / sqrt(1 + 0.5*0.01) = 0.997509; Phi(z) = 0.84074
    assert probabilistic_sharpe_ratio(0.1, 101, 0.0, 3.0) == pytest.approx(0.84074, abs=2e-5)


def test_psr_with_skew_and_fat_tails():
    # SR=0.1, T=101, skew=-1, kurt=7, SR*=0.02:
    # denom = sqrt(1 + 0.1 + 1.5*0.01) = sqrt(1.115) = 1.055936 ; z = 0.08*10/1.055936 = 0.757622
    assert probabilistic_sharpe_ratio(0.1, 101, -1.0, 7.0, 0.02) == pytest.approx(0.77566, abs=2e-5)


def test_psr_is_half_at_the_benchmark():
    assert probabilistic_sharpe_ratio(0.05, 1000, -0.5, 5.0, 0.05) == pytest.approx(0.5)


def test_expected_max_sharpe():
    # N=10, V=0.01: PhiInv(0.9) = 1.281552 ; PhiInv(1 - 1/(10e)) = PhiInv(0.963212) = 1.789224
    # SR* = 0.1 * (0.422784*1.281552 + 0.577216*1.789224) = 0.1 * 1.574589 = 0.157459
    assert expected_max_sharpe(0.01, 10) == pytest.approx(0.157459, abs=2e-5)
    assert expected_max_sharpe(0.01, 1) == 0.0
    assert expected_max_sharpe(0.01, 100) > expected_max_sharpe(0.01, 10)


def test_expected_max_sharpe_independent_implementation():
    nd, g = NormalDist(), 0.5772156649015329
    for var, n in [(0.004, 25), (0.0001, 300)]:
        want = math.sqrt(var) * ((1 - g) * nd.inv_cdf(1 - 1 / n) + g * nd.inv_cdf(1 - 1 / (n * math.e)))
        assert expected_max_sharpe(var, n) == pytest.approx(want, rel=1e-9)


def test_moments_raw_kurtosis_convention():
    r = np.array([1.0, -1.0] * 50)                    # two-point distribution: skew 0, raw kurtosis 1
    _, n, skew, kurt = moments(r)
    assert (n, skew, kurt) == (100, 0.0, 1.0)
    assert moments(np.random.default_rng(0).normal(size=200_000))[3] == pytest.approx(3.0, abs=0.05)


def test_dsr_equals_psr_at_expected_max_and_falls_with_more_trials():
    rng = np.random.default_rng(2)
    r = rng.normal(0.05, 1.0, 2000)
    trial_sr = rng.normal(0, 0.03, 20)
    out = deflated_sharpe_ratio(r, trial_sr)
    sr, n, skew, kurt = moments(r)
    star = expected_max_sharpe(trial_sr.var(ddof=1), 20)
    assert out["dsr"] == pytest.approx(probabilistic_sharpe_ratio(sr, n, skew, kurt, star))
    assert deflated_sharpe_ratio(r, trial_sr, n_trials=200)["dsr"] < out["dsr"]
    assert out["dsr"] < probabilistic_sharpe_ratio(sr, n, skew, kurt, 0.0)


def test_effective_n_limits():
    rng = np.random.default_rng(3)
    independent = pd.DataFrame(rng.normal(size=(5000, 10)))
    eff = effective_n_trials(independent)
    assert eff["n_eff_avg_corr"] == pytest.approx(10, abs=0.5) and eff["n_eff_eigen"] == pytest.approx(10, abs=0.5)
    base = rng.normal(size=5000)
    clones = pd.DataFrame({i: base + rng.normal(0, 0.01, 5000) for i in range(10)})
    eff = effective_n_trials(clones)
    assert eff["n_eff_avg_corr"] < 1.1 and eff["n_eff_eigen"] < 1.1


def test_pbo_noise_is_near_half_and_real_edge_is_near_zero():
    rng = np.random.default_rng(4)
    noise = pd.DataFrame(rng.normal(0, 0.01, (2000, 20)))
    assert 0.3 < probability_of_backtest_overfitting(noise, 10)["pbo"] < 0.7
    edge = noise.copy()
    edge[0] += 0.002                                  # one trial with a large true Sharpe
    assert probability_of_backtest_overfitting(edge, 10)["pbo"] < 0.05
