"""Independent Gaussian complex components, followed by explicit magnitude."""

from numbers import Integral

import numpy as np


def add_complex_noise(signal, *, repetitions, sigma, rng):
    """Return ``(b_value, repetition, *spatial)`` complex measurements.

    ``sigma`` is the standard deviation of EACH real/imaginary component.
    Magnitude formation belongs after this step. This single-channel model
    does not simulate multicoil reconstruction or spatial noise covariance.
    """
    values = np.asarray(signal, dtype=complex)
    if values.ndim < 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError('signal must be finite with a nonempty b-value axis')
    if isinstance(repetitions, bool) or not isinstance(repetitions, Integral) or repetitions <= 0:
        raise ValueError('repetitions must be a positive integer')
    if not np.isscalar(sigma) or not np.isfinite(sigma) or sigma < 0:
        raise ValueError('sigma must be a finite nonnegative scalar')
    if not isinstance(rng, np.random.Generator):
        raise ValueError('an explicit NumPy Generator is required')
    shape = (values.shape[0], repetitions) + values.shape[1:]
    real = rng.normal(0, sigma, size=shape)
    imaginary = rng.normal(0, sigma, size=shape)
    return values[:, None, ...] + real + 1j * imaginary
