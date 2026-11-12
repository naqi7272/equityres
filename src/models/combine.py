"""Four ways to combine signals into one score. All share fit(ds, train_idx) and
predict(ds, idx), where the indices are date positions. Nothing a model learns
may come from outside train_idx."""
from __future__ import annotations

import numpy as np

from src.models.dataset import Dataset


def _train_rows(ds: Dataset, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ok = ds.universe[train_idx] & np.isfinite(ds.y[train_idx])
    return ds.z[train_idx][ok], ds.y[train_idx][ok]


class EqualWeight:
    """Average of the z-scores. Nothing is fitted, so it cannot overfit."""

    def fit(self, ds: Dataset, train_idx: np.ndarray) -> "EqualWeight":
        return self

    def predict(self, ds: Dataset, idx: np.ndarray) -> np.ndarray:
        return ds.z[idx].mean(axis=-1)


class ICWeighted:
    """Weights proportional to each signal's mean rank IC on the training dates.

    Negative weights are set to zero: the signs were fixed a priori, and a
    signal with negative training IC is dropped rather than flipped.
    """

    def fit(self, ds: Dataset, train_idx: np.ndarray) -> "ICWeighted":
        w = np.clip(np.nanmean(ds.daily_ic[train_idx], axis=0), 0.0, None)
        self.weights_ = w / w.sum() if w.sum() > 0 else np.full(len(w), 1.0 / len(w))
        return self

    def predict(self, ds: Dataset, idx: np.ndarray) -> np.ndarray:
        return ds.z[idx] @ self.weights_.astype(np.float32)


class RidgeModel:
    """Ridge regression of the target on the z-scores, no intercept, closed form.

    Penalty = lam * n_rows, so `lam` means the same thing for any sample size.
    """

    def __init__(self, lam: float):
        self.lam = lam

    def fit(self, ds: Dataset, train_idx: np.ndarray) -> "RidgeModel":
        x, y = _train_rows(ds, train_idx)
        x = x.astype(np.float64)
        gram = x.T @ x + self.lam * len(x) * np.eye(x.shape[1])
        self.coef_ = np.linalg.solve(gram, x.T @ y.astype(np.float64))
        return self

    def predict(self, ds: Dataset, idx: np.ndarray) -> np.ndarray:
        return ds.z[idx] @ self.coef_.astype(np.float32)


class LightGBMModel:
    """Gradient-boosted shallow trees with strong regularization."""

    def __init__(self, n_estimators: int, max_depth: int, seed: int, train_every: int = 1, **fixed):
        self.params = dict(n_estimators=n_estimators, max_depth=max_depth, num_leaves=2 ** max_depth,
                           random_state=seed, deterministic=True, force_col_wise=True, verbose=-1, **fixed)
        self.train_every = train_every

    def fit(self, ds: Dataset, train_idx: np.ndarray) -> "LightGBMModel":
        from lightgbm import LGBMRegressor

        x, y = _train_rows(ds, train_idx[:: self.train_every])
        self.model_ = LGBMRegressor(**self.params).fit(x, y)
        return self

    def predict(self, ds: Dataset, idx: np.ndarray) -> np.ndarray:
        block = ds.z[idx]
        return self.model_.predict(block.reshape(-1, block.shape[-1])).reshape(block.shape[:2])


def make_model(family: str, params: dict, cfg: dict):
    """Build a model from its family name and grid parameters."""
    if family == "equal_weight":
        return EqualWeight()
    if family == "ic_weighted":
        return ICWeighted()
    if family == "ridge":
        return RidgeModel(**params)
    if family == "lightgbm":
        return LightGBMModel(seed=cfg["seed"], **params, **cfg["models"]["lightgbm_fixed"])
    raise ValueError(f"unknown model family {family}")
