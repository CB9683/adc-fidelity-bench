"""Monoexponential tissue signals; ADC is mm^2/s and b-values are s/mm^2."""

import numpy as np


def monoexponential_signal(b_values, adc, s0=1.0):
    """Return noiseless signals with shape ``(b_value, *spatial)``.

    Zero amplitudes model background. ADC and amplitude may broadcast, but
    neither negative nor nonfinite generator parameters are accepted.
    """
    b = np.asarray(b_values, dtype=float)
    if (b.ndim != 1 or b.size == 0 or not np.isfinite(b).all()
            or np.any(b < 0) or np.any(np.diff(b) <= 0)):
        raise ValueError('b-values must be a nonempty, finite, increasing nonnegative vector')
    d, amplitude = np.broadcast_arrays(np.asarray(adc, dtype=float), np.asarray(s0, dtype=float))
    if (d.size == 0 or not np.isfinite(d).all() or not np.isfinite(amplitude).all()
            or np.any(d < 0) or np.any(amplitude < 0)):
        raise ValueError('ADC and S0 must be finite and nonnegative')
    return amplitude[None, ...] * np.exp(-b.reshape((-1,) + (1,) * d.ndim) * d[None, ...])


def mix_tissue_signals(fractions, tissue_adc, tissue_s0, b_values):
    """Mix signals by tissue fractions, never by averaging tissue ADC.

    Fractions have tissue first and sum to one at each spatial location;
    include an explicit zero-amplitude background tissue where needed.
    """
    weights = np.asarray(fractions, dtype=float)
    d = np.asarray(tissue_adc, dtype=float)
    amplitude = np.asarray(tissue_s0, dtype=float)
    if (weights.ndim < 1 or weights.size == 0 or not np.isfinite(weights).all()
            or np.any(weights < 0) or not np.allclose(weights.sum(axis=0), 1.0, rtol=0, atol=1e-8)):
        raise ValueError('finite nonnegative fractions must sum to one at each location')
    if d.ndim != 1 or amplitude.shape != d.shape or len(d) != weights.shape[0]:
        raise ValueError('one ADC and S0 value is required for each tissue')
    tissue_signals = monoexponential_signal(b_values, d, amplitude)
    return np.tensordot(tissue_signals, weights, axes=(1, 0))
