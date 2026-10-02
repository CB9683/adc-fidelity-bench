"""Log-linear and known-noise Rician monoexponential ADC fitting."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import i0e, i1e


@dataclass(frozen=True)
class ADCResult:
    """Spatial arrays; only ``valid`` ADC entries may enter evaluation.

    ``converged`` records numerical success separately from identifiability.
    An analytically certified zero-amplitude Rician solution has
    converged=True, s0=0, valid=False,
    and adc=NaN because its diffusivity is unidentified. Rician fits whose
    likelihood has only an infinite-ADC supremum are likewise invalid even
    if an optimizer reports numerical convergence. Failed optimization
    retains a finite fitted s0, when available, but never a usable ADC.
    """

    adc: np.ndarray
    valid: np.ndarray
    s0: np.ndarray
    converged: np.ndarray


def _real_array(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real-valued")
    try:
        return np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array") from exc


def _b_values(value, length):
    b = _real_array(value, "b_values")
    if (b.ndim != 1 or b.size < 2 or b.size != length
            or not np.all(np.isfinite(b)) or np.any(b < 0)
            or np.any(np.diff(b) <= 0)):
        raise ValueError("b_values must match the b axis and be finite, nonnegative, strictly increasing, with length >= 2")
    return b


def _empty_result(shape):
    return ADCResult(np.full(shape, np.nan), np.zeros(shape, dtype=bool),
                     np.full(shape, np.nan), np.zeros(shape, dtype=bool))


def fit_log_linear_adc(signals, b_values):
    """Fit log(S)=log(S0)-b*ADC without constraining or clipping ADC.

    Signals have shape (B, *spatial), b-values are in s/mm², and returned ADC
    is in mm²/s. A voxel requires a positive finite signal at every b-value.
    """
    signal = _real_array(signals, "signals")
    if signal.ndim < 1:
        raise ValueError("signals require a leading b-value axis")
    b = _b_values(b_values, signal.shape[0])
    shape = signal.shape[1:]
    result = _empty_result(shape)
    spatial_size = int(np.prod(shape, dtype=int))
    flat = signal.reshape((b.size, spatial_size))
    eligible = np.all(np.isfinite(flat) & (flat > 0), axis=0)
    if not np.any(eligible):
        return result
    # Normalize b before centering to avoid squaring large dimensional values.
    b_scale = b[-1]
    normalized_b = b / b_scale
    centered_b = normalized_b - normalized_b.mean()
    log_signal = np.log(flat[:, eligible])
    slope = np.sum(centered_b[:, None] * log_signal, axis=0) / np.sum(centered_b**2)
    adc = -slope / b_scale
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        s0 = np.exp(log_signal.mean(axis=0) - normalized_b.mean() * slope)
    finite = np.isfinite(adc) & np.isfinite(s0) & (s0 > 0)
    locations = np.flatnonzero(eligible)[finite]
    result.adc.reshape(-1)[locations] = adc[finite]
    result.s0.reshape(-1)[locations] = s0[finite]
    result.valid.reshape(-1)[locations] = True
    result.converged.reshape(-1)[locations] = True
    return result


def fit_rician_adc(magnitude, b_values, *, sigma, maxiter=200):
    """Fit nonnegative S0/ADC to all (B, R, *spatial) magnitude observations.

    Sigma is the known positive standard deviation of each independent
    complex Gaussian noise component, in the same units as the observations.
    This comparator assumes simulated noise metadata; it does not estimate
    sigma. Initialization uses observed mean magnitudes only. No reference
    signal, S0, or ADC enters the estimator.

    The scaled-Bessel likelihood is equivalent to the Rician density after
    dropping parameter-independent terms. Optimization scales ADC by max(b)
    and amplitude by the largest observed magnitude for numerical conditioning.

    If every higher-b independent Rice amplitude MLE is zero, ADC has no
    identifiable finite maximum: it is unbounded when the lowest-b amplitude
    is positive, and unidentified when all amplitudes are zero. Such ADCs
    are excluded independently of numerical convergence. For two b-values
    this exactly detects the zero-high-b boundary. For more b-values it is
    a sufficient boundary check, not a general identifiability guarantee.

    A global S0=0 optimum is certified when every increasing-b prefix has
    mean squared magnitude <= 2*sigma². This criterion is exact for two
    b-values and sufficient for more b-values. Certified null fits return
    s0=0 and converged=True analytically, with unidentified ADC excluded.
    A numerical candidate is also excluded if its fitted-ADC weighted moment
    score certifies conditional S0=0. These checks do not guarantee global
    optimization or general multi-b identifiability.
    """
    observation = _real_array(magnitude, "magnitude")
    if observation.ndim < 2 or observation.shape[1] < 1:
        raise ValueError("magnitude requires b-value and nonempty repetition axes")
    b = _b_values(b_values, observation.shape[0])
    noise = _real_array(sigma, "sigma")
    if noise.ndim != 0 or not np.isfinite(noise) or noise <= 0:
        raise ValueError("sigma must be a finite positive scalar")
    if isinstance(maxiter, (bool, np.bool_)) or not isinstance(maxiter, (int, np.integer)) or maxiter < 1:
        raise ValueError("maxiter must be a positive integer")
    shape = observation.shape[2:]
    result = _empty_result(shape)
    spatial_size = int(np.prod(shape, dtype=int))
    flat = observation.reshape((b.size, observation.shape[1], spatial_size))
    eligible = np.all(np.isfinite(flat) & (flat >= 0), axis=(0, 1))
    b_scale = b[-1]
    normalized_b = (b / b_scale)[:, None]
    for location in np.flatnonzero(eligible):
        data = flat[:, :, location]
        amplitude_scale = float(np.max(data))
        if amplitude_scale == 0:
            result.s0.reshape(-1)[location] = 0
            result.converged.reshape(-1)[location] = True
            continue
        y = data / amplitude_scale
        with np.errstate(over="ignore", under="ignore"):
            variance = float(np.square(noise / amplitude_scale))
        if not np.isfinite(variance) or variance <= 0:
            continue
        half_moments = 0.5 * np.mean(y**2, axis=1)
        # log(I0(z)) <= z²/4 bounds the NLL increase from S0=0 below by a
        # positive multiple of -S0²*sum(c_b*exp(-2*b*ADC)), where
        # c_b=mean(y_b²)/2-variance. Nonpositive prefix sums make this
        # weighted sum nonpositive for every ADC>=0 by Abel summation.
        # The first branch also avoids summing extremely large variances.
        null_amplitude = (np.all(half_moments <= variance)
                          or np.all(np.cumsum(half_moments - variance) <= 0))
        if null_amplitude:
            result.s0.reshape(-1)[location] = 0
            result.converged.reshape(-1)[location] = True
            continue
        # The shared Rice amplitude MLE is zero iff mean(y²) <= 2*variance.
        # Its score uses I1(z)/I0(z) < z/2: below this moment threshold the
        # negative log likelihood increases from amplitude zero. If all
        # higher-b amplitudes have that maximum, sending ADC to infinity
        # minimizes them simultaneously without changing the lowest-b fit.
        zero_high_b_mles = np.all(half_moments[1:] <= variance)
        means = y.mean(axis=1)
        start = fit_log_linear_adc(np.maximum(means, np.finfo(float).tiny), b / b_scale)
        initial = [float(start.s0) if start.valid else 1.0,
                   max(float(start.adc), 0.0) if start.valid else 0.0]

        def likelihood(parameters):
            amplitude, diffusivity = parameters
            decay = np.exp(-normalized_b * diffusivity)
            signal = amplitude * decay
            argument = y * signal / variance
            scaled_i0 = i0e(argument)
            # Rewriting quadratic minus z avoids large high-SNR cancellation.
            objective = np.mean(0.5 * (y - signal)**2 - variance * np.log(scaled_i0))
            residual = signal - y * (i1e(argument) / scaled_i0)
            gradient = np.array([np.mean(residual * decay),
                                 np.mean(residual * (-normalized_b * signal))])
            return objective, gradient

        fitted = minimize(likelihood, initial, method="L-BFGS-B", jac=True,
                          bounds=((0, None), (0, None)),
                          options={"maxiter": int(maxiter), "ftol": 1e-12, "gtol": 1e-9})
        amplitude, diffusivity = fitted.x
        with np.errstate(over="ignore", invalid="ignore"):
            fitted_s0 = amplitude * amplitude_scale
            fitted_adc = diffusivity / b_scale
        finite = np.isfinite(fitted_s0) and np.isfinite(fitted_adc) and np.isfinite(fitted.fun)
        if np.isfinite(fitted_s0):
            result.s0.reshape(-1)[location] = fitted_s0
        converged = bool(fitted.success and finite)
        result.converged.reshape(-1)[location] = converged
        if converged and fitted_s0 > 0 and not zero_high_b_mles:
            # At fixed D, nonpositive weighted moment score certifies a
            # zero amplitude maximum even when the global prefix test fails.
            weights = np.exp(-2 * (b / b_scale) * diffusivity)
            if np.sum((half_moments - variance) * weights) > 0:
                result.adc.reshape(-1)[location] = fitted_adc
                result.valid.reshape(-1)[location] = True
    return result
