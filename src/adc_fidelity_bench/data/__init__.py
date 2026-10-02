"""Anatomy-level data partitioning."""

from .splits import grouped_split, validate_split_integrity

__all__ = ["grouped_split", "validate_split_integrity"]
