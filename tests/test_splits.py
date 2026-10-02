"""Anatomy grouping is fixed before any derived examples are generated."""

import numpy as np
import pytest

from adc_fidelity_bench.data.splits import grouped_split, validate_split_integrity


SPLIT_NAMES = ("train", "validation", "calibration", "test")


def test_grouped_split_is_reproducible_independent_of_input_order():
    anatomy_ids = [f"anatomy_{index:02d}" for index in range(12)]
    counts = dict(zip(SPLIT_NAMES, [4, 2, 2, 4]))

    splits = grouped_split(anatomy_ids, counts, seed=41)

    assert splits == grouped_split(list(reversed(anatomy_ids)), counts, seed=41)
    assert splits != grouped_split(anatomy_ids, counts, seed=42)
    assert list(splits) == list(SPLIT_NAMES)
    assert {key: len(value) for key, value in splits.items()} == counts
    assert set(sum(splits.values(), [])) == set(anatomy_ids)
    validate_split_integrity(splits)


def test_grouped_split_allows_zero_counts_and_numpy_integer_counts():
    splits = grouped_split(["a"], dict(zip(SPLIT_NAMES, [np.int64(1), 0, 0, 0])), seed=0)
    assert splits == {"train": ["a"], "validation": [], "calibration": [], "test": []}


def test_grouped_split_accepts_a_fixed_numpy_integer_seed():
    ids = ["a", "b", "c"]
    counts = dict(zip(SPLIT_NAMES, [1, 0, 1, 1]))
    assert grouped_split(ids, counts, seed=np.int64(7)) == grouped_split(ids, counts, seed=7)


@pytest.mark.parametrize("seed", [None, True, False, np.bool_(True), -1, 7.0, [7]])
def test_grouped_split_rejects_seeds_without_a_fixed_nonnegative_integer(seed):
    counts = dict(zip(SPLIT_NAMES, [1, 0, 0, 0]))
    with pytest.raises(ValueError, match="seed|Seed"):
        grouped_split(["a"], counts, seed=seed)


@pytest.mark.parametrize("ids", [["a", "a"], [""], ["   "], [1], [None]])
def test_grouped_split_rejects_duplicate_or_invalid_anatomy_ids(ids):
    counts = dict(zip(SPLIT_NAMES, [len(ids), 0, 0, 0]))
    with pytest.raises(ValueError):
        grouped_split(ids, counts, seed=1)


@pytest.mark.parametrize("counts", [
    {"train": 1, "validation": 0, "calibration": 0},
    {"train": 1, "validation": 0, "calibration": 0, "test": 0, "other": 0},
    dict(zip(SPLIT_NAMES, [-1, 1, 0, 1])),
    dict(zip(SPLIT_NAMES, [True, 0, 0, 0])),
    dict(zip(SPLIT_NAMES, [1.0, 0, 0, 0])),
    dict(zip(SPLIT_NAMES, [0, 0, 0, 0])),
])
def test_grouped_split_rejects_invalid_counts(counts):
    with pytest.raises(ValueError):
        grouped_split(["a"], counts, seed=1)


@pytest.mark.parametrize("splits", [
    {"train": ["a", "a"], "validation": [], "calibration": [], "test": []},
    {"train": ["a"], "validation": [], "calibration": [], "test": ["a"]},
    {"train": [1], "validation": [], "calibration": [], "test": []},
    {"train": [""], "validation": [], "calibration": [], "test": []},
    {"train": [], "validation": [], "calibration": []},
    {"train": [], "validation": [], "calibration": [], "test": [], "other": []},
    {"train": "abc", "validation": [], "calibration": [], "test": []},
])
def test_validate_split_integrity_rejects_malformed_or_leaking_assignments(splits):
    with pytest.raises(ValueError):
        validate_split_integrity(splits)
