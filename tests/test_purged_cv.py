"""Purge test: no training label window may overlap a test label window, and
the embargo after each test block must be empty of training dates."""
from math import comb

import numpy as np
import pytest

from src.timing import ENTRY_LAG, label_span
from src.validation.cpcv import CombinatorialPurgedCV
from src.validation.purged_kfold import PurgedKFold, contiguous_blocks

N, H, EMBARGO = 500, 5, 7


def _label_days(i: int, h: int) -> set[int]:
    """Days whose close enters the label of date i: i+lag .. i+lag+h."""
    return set(range(i + ENTRY_LAG, i + label_span(h) + 1))


def _check(train_idx, test_idx):
    assert len(np.intersect1d(train_idx, test_idx)) == 0
    test_days = set().union(*(_label_days(j, H) for j in test_idx))
    for i in train_idx:
        assert not (_label_days(i, H) & test_days), f"train date {i} overlaps the test labels"
    train = set(train_idx.tolist())
    for _, end in contiguous_blocks(test_idx):
        assert not (train & set(range(end + 1, end + H + EMBARGO + 1))), "embargo violated"


def test_purged_kfold_has_no_overlap_and_respects_embargo():
    splitter = PurgedKFold(5, H, EMBARGO)
    tested = []
    for train_idx, test_idx in splitter.split(N):
        _check(train_idx, test_idx)
        assert len(train_idx) > N // 2
        tested.append(test_idx)
    assert np.array_equal(np.sort(np.concatenate(tested)), np.arange(N)), "every date is tested exactly once"


def test_purging_removes_exactly_the_expected_dates():
    train_idx, test_idx = list(PurgedKFold(5, H, EMBARGO).split(N))[2]       # test block 200..299
    expected = np.r_[np.arange(0, 200 - H), np.arange(299 + H + EMBARGO + 1, N)]
    assert np.array_equal(train_idx, expected)


def test_cpcv_has_no_overlap_and_right_counts():
    cpcv = CombinatorialPurgedCV(6, 2, H, EMBARGO)
    splits = list(cpcv.split(N))
    assert len(splits) == cpcv.n_splits == comb(6, 2) == 15
    for train_idx, test_idx, _ in splits:
        _check(train_idx, test_idx)


def test_cpcv_paths_use_each_split_group_once():
    cpcv = CombinatorialPurgedCV(6, 2, H, EMBARGO)
    paths = cpcv.paths()
    assert len(paths) == cpcv.n_paths == 5
    combos = [combo for _, _, combo in cpcv.split(N)]
    used = set()
    for path in paths:
        for g, split_id in enumerate(path):
            assert g in combos[split_id], "a path may only take a group from a split that tests it"
            used.add((split_id, g))
    assert len(used) == 15 * 2, "every (split, test group) pair is used on exactly one path"


def test_invalid_arguments():
    with pytest.raises(ValueError):
        PurgedKFold(1, H)
    with pytest.raises(ValueError):
        CombinatorialPurgedCV(4, 4, H)
