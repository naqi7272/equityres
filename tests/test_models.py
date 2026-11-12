"""Combination models: leak-free fitting and recovery of a planted combination."""
import numpy as np
import pandas as pd

from src.models.combine import EqualWeight, ICWeighted, RidgeModel
from src.models.cv import cpcv_scores, kfold_scores
from src.models.dataset import build_dataset
from src.validation.cpcv import CombinatorialPurgedCV
from src.validation.purged_kfold import PurgedKFold


def _dataset(n_days=300, n_names=120, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n_days)
    good, noise = (pd.DataFrame(rng.normal(size=(n_days, n_names)), index=idx) for _ in range(2))
    label = 0.2 * good + rng.normal(size=(n_days, n_names))
    tradable = pd.DataFrame(True, index=idx, columns=label.columns)
    return build_dataset({"good": good, "noise": noise}, label, tradable)


def test_ridge_and_ic_weights_find_the_real_signal():
    ds = _dataset()
    train = np.arange(200)
    ridge = RidgeModel(lam=0.1).fit(ds, train)
    assert ridge.coef_[0] > 0.1 and abs(ridge.coef_[1]) < 0.03
    icw = ICWeighted().fit(ds, train)
    assert icw.weights_[0] > 0.9


def test_models_only_read_training_dates():
    """Changing labels and ICs outside the training dates must not change the fit."""
    ds = _dataset()
    train = np.arange(50, 150)
    before = (RidgeModel(0.1).fit(ds, train).coef_.copy(), ICWeighted().fit(ds, train).weights_.copy())
    outside = np.setdiff1d(np.arange(len(ds.dates)), train)
    ds.y[outside] = 123.0
    ds.daily_ic[outside] = -1.0
    after = (RidgeModel(0.1).fit(ds, train).coef_, ICWeighted().fit(ds, train).weights_)
    assert np.array_equal(before[0], after[0]) and np.array_equal(before[1], after[1])


def test_kfold_scores_cover_all_dates_and_cpcv_paths_are_complete():
    ds = _dataset()
    scores = kfold_scores(EqualWeight(), ds, PurgedKFold(5, 5, 5))
    assert np.isfinite(scores).all()
    cpcv = CombinatorialPurgedCV(6, 2, 5, 5)
    splits, paths = cpcv_scores(RidgeModel(0.1), ds, cpcv)
    assert len(splits) == 15 and len(paths) == 5
    assert all(np.isfinite(p).all() for p in paths)
    assert np.isfinite(splits[0]).any(axis=1).mean() == 2 / 6
