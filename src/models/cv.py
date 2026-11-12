"""Run combination models under purged K-fold and CPCV."""
from __future__ import annotations

import numpy as np

from src.models.dataset import Dataset
from src.validation.cpcv import CombinatorialPurgedCV
from src.validation.purged_kfold import PurgedKFold


def _masked(ds: Dataset, idx: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.where(ds.universe[idx], pred, np.nan)


def kfold_scores(model, ds: Dataset, splitter: PurgedKFold) -> np.ndarray:
    """Out-of-sample score for every date: each date is predicted by the model
    trained on the purged folds that exclude it."""
    scores = np.full(ds.universe.shape, np.nan)
    for train_idx, test_idx in splitter.split(len(ds.dates)):
        scores[test_idx] = _masked(ds, test_idx, model.fit(ds, train_idx).predict(ds, test_idx))
    return scores


def cpcv_scores(model, ds: Dataset, cpcv: CombinatorialPurgedCV) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return (per-split score panels, per-path score panels).

    A split panel holds scores on that split's test dates only. A path panel
    covers all dates, stitched from the splits listed in cpcv.paths().
    """
    n = len(ds.dates)
    groups = cpcv.groups(n)
    split_panels = []
    for train_idx, test_idx, _ in cpcv.split(n):
        panel = np.full(ds.universe.shape, np.nan)
        panel[test_idx] = _masked(ds, test_idx, model.fit(ds, train_idx).predict(ds, test_idx))
        split_panels.append(panel)
    path_panels = []
    for path in cpcv.paths():
        panel = np.full(ds.universe.shape, np.nan)
        for g, split_id in enumerate(path):
            panel[groups[g]] = split_panels[split_id][groups[g]]
        path_panels.append(panel)
    return split_panels, path_panels
