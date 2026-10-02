"""A documented box-response acquisition approximation in signal space."""

from numbers import Integral

import numpy as np


def block_average(data, factor):
    """Average non-overlapping blocks in the last two axes without padding.

    This is a box point-spread/resampling approximation, not a scanner model.
    Leading axes (b-value, direction, or repetition) and complex phase are preserved.
    """
    values = np.asarray(data, dtype=complex if np.iscomplexobj(data) else float)
    if isinstance(factor, bool) or not isinstance(factor, Integral) or factor <= 0:
        raise ValueError('factor must be a positive integer')
    if values.ndim < 2 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError('data require finite, nonempty spatial axes')
    height, width = values.shape[-2:]
    if height % factor or width % factor:
        raise ValueError('spatial dimensions must be divisible by factor; padding is not implicit')
    shape = values.shape[:-2] + (height // factor, factor, width // factor, factor)
    return values.reshape(shape).mean(axis=(-3, -1))
