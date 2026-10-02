"""Reproducible partitions that keep each anatomy in a single data split."""

from collections.abc import Iterable, Mapping
from numbers import Integral

import numpy as np


SPLIT_NAMES = ("train", "validation", "calibration", "test")


def _validated_ids(anatomy_ids: Iterable[str]) -> list[str]:
    if isinstance(anatomy_ids, (str, bytes)):
        raise ValueError("Anatomy IDs must be a collection of nonempty strings.")
    try:
        ids = list(anatomy_ids)
    except TypeError as error:
        raise ValueError("Anatomy IDs must be a collection of strings.") from error
    if any(not isinstance(identifier, str) or not identifier.strip() for identifier in ids):
        raise ValueError("Each anatomy ID must be a nonempty string.")
    if len(set(ids)) != len(ids):
        raise ValueError("Anatomy IDs must not contain duplicates.")
    return ids


def _validate_keys(mapping: Mapping) -> None:
    if not isinstance(mapping, Mapping) or set(mapping) != set(SPLIT_NAMES):
        raise ValueError(f"Split keys must be exactly {SPLIT_NAMES}.")


def grouped_split(
    anatomy_ids: Iterable[str], counts: Mapping[str, int], seed: int
) -> dict[str, list[str]]:
    """Allocate whole anatomy IDs using one seeded shuffle of sorted IDs.

    Derive slices, noise realizations, and focal perturbations only after this
    assignment; every example from an anatomy must inherit its assigned split.
    Counts refer to anatomies, rather than images or derived examples.
    """
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("Seed must be a fixed nonnegative integer, excluding booleans.")
    ids = _validated_ids(anatomy_ids)
    _validate_keys(counts)
    if any(isinstance(count, bool) or not isinstance(count, Integral) or count < 0
           for count in counts.values()):
        raise ValueError("Split counts must be nonnegative integers, excluding booleans.")
    if sum(counts.values()) != len(ids):
        raise ValueError("Split counts must sum to the number of anatomy IDs.")

    shuffled = np.asarray(sorted(ids), dtype=object)
    np.random.default_rng(seed).shuffle(shuffled)
    splits = {}
    start = 0
    for name in SPLIT_NAMES:
        end = start + int(counts[name])
        splits[name] = shuffled[start:end].tolist()
        start = end
    return splits


def validate_split_integrity(splits: Mapping[str, Iterable[str]]) -> None:
    """Reject malformed IDs and within-split or cross-split anatomy overlap."""
    _validate_keys(splits)
    assigned = set()
    for name in SPLIT_NAMES:
        ids = _validated_ids(splits[name])
        if assigned.intersection(ids):
            raise ValueError("Anatomy IDs must not appear in multiple splits.")
        assigned.update(ids)
