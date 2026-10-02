"""Known-answer and failure-reporting tests for conventional ADC fits."""

import importlib

import numpy as np
import pytest


def _estimators():
    try:
        return importlib.import_module("adc_fidelity_bench.baselines")
    except ImportError:
        pytest.fail("Conventional ADC estimators are not implemented yet.", pytrace=False)


def _signals(b_values, adc, s0):
    adc, s0 = np.broadcast_arrays(adc, s0)
    b = np.asarray(b_values).reshape((-1,) + (1,) * adc.ndim)
    return s0 * np.exp(-b * adc)


@pytest.mark.parametrize("shape", [(), (3,), (2, 2)])
def test_log_fit_recovers_adc_units_and_spatial_shape(shape):
    adc = np.full(shape, 0.0008)
    s0 = np.full(shape, 3.7)
    b_values = [0, 200, 800, 1200]
    result = _estimators().fit_log_linear_adc(_signals(b_values, adc, s0), b_values)
    for field in (result.adc, result.s0, result.valid, result.converged):
        assert field.shape == shape
    np.testing.assert_allclose(result.adc, adc, rtol=1e-12)
    np.testing.assert_allclose(result.s0, s0, rtol=1e-12)
    assert np.all(result.valid)
    assert np.all(result.converged)


def test_log_fit_extrapolates_s0_without_b_zero():
    b_values = [250, 600, 1000]
    result = _estimators().fit_log_linear_adc(_signals(b_values, 0.0012, 125), b_values)
    assert result.adc == pytest.approx(0.0012)
    assert result.s0 == pytest.approx(125)


def test_log_fit_retains_negative_adc_for_reversed_slope():
    result = _estimators().fit_log_linear_adc([1, 2], [0, 1000])
    assert result.valid
    assert result.adc == pytest.approx(-np.log(2) / 1000)


def test_log_fit_uses_all_b_values_for_noisy_data():
    signals = [1.1, 0.9, 0.7, 0.72]
    b_values = [0, 250, 600, 1000]
    slope, intercept = np.polyfit(b_values, np.log(signals), 1)
    result = _estimators().fit_log_linear_adc(signals, b_values)
    assert result.adc == pytest.approx(-slope)
    assert result.s0 == pytest.approx(np.exp(intercept))


def test_log_fit_excludes_only_invalid_voxels_without_clipping():
    signals = np.array([[1, 1, 1, 1, 1, 1], [0.5, 0, -1, np.nan, np.inf, 2]])
    result = _estimators().fit_log_linear_adc(signals, [0, 1000])
    np.testing.assert_array_equal(result.valid, [True, False, False, False, False, True])
    np.testing.assert_array_equal(result.converged, result.valid)
    assert np.all(np.isnan(result.adc[1:5]))
    assert np.all(np.isnan(result.s0[1:5]))
    assert result.adc[-1] < 0


@pytest.mark.parametrize("b_values", [
    [0], [0, 1000, 1500], [[0, 1000]], [-1, 1000], [0, np.nan],
    [0, np.inf], [1000, 0], [0, 0], [0, 1000j],
])
def test_estimators_reject_invalid_b_values(b_values):
    estimators = _estimators()
    with pytest.raises(ValueError):
        estimators.fit_log_linear_adc([1, 0.5], b_values)
    with pytest.raises(ValueError):
        estimators.fit_rician_adc([[1, 1], [0.5, 0.5]], b_values, sigma=0.1)


@pytest.mark.parametrize("signals", [1, [1j, 0.5]])
def test_log_fit_rejects_missing_b_axis_and_complex_input(signals):
    with pytest.raises(ValueError):
        _estimators().fit_log_linear_adc(signals, [0, 1000])


@pytest.mark.parametrize("shape", [(), (3,), (2, 2)])
def test_rician_fit_recovers_high_snr_signal_without_bessel_overflow(shape):
    adc = np.full(shape, 0.0008)
    s0 = np.full(shape, 3.7)
    b_values = [0, 500, 1000]
    signal = _signals(b_values, adc, s0)
    noise_shape = (1, 4) + (1,) * len(shape)
    real_noise = np.array([-1, 1, -1, 1]).reshape(noise_shape) * 0.002
    imag_noise = np.array([-1, -1, 1, 1]).reshape(noise_shape) * 0.002
    magnitude = np.abs(signal[:, None] + real_noise + 1j * imag_noise)
    result = _estimators().fit_rician_adc(magnitude, b_values, sigma=0.002)
    for field in (result.adc, result.s0, result.valid, result.converged):
        assert field.shape == shape
    assert np.all(result.valid)
    assert np.all(result.converged)
    np.testing.assert_allclose(result.adc, adc, rtol=1e-4)
    np.testing.assert_allclose(result.s0, s0, rtol=1e-4)


def test_rician_fit_is_invariant_to_signal_and_sigma_amplitude_units():
    magnitude = np.array([[1.02, 0.95, 1.1], [0.42, 0.55, 0.48], [0.2, 0.25, 0.3]])
    estimators = _estimators()
    result = estimators.fit_rician_adc(magnitude, [0, 800, 1400], sigma=0.08)
    scaled = estimators.fit_rician_adc(magnitude * 1e4, [0, 800, 1400], sigma=800)
    assert result.valid and scaled.valid
    assert scaled.adc == pytest.approx(result.adc, rel=1e-8)
    assert scaled.s0 == pytest.approx(result.s0 * 1e4, rel=1e-8)


def test_rician_fit_constrains_reversed_slope_to_zero_adc():
    result = _estimators().fit_rician_adc([[1, 1.1], [2, 1.9]], [0, 1000], sigma=0.1)
    assert result.valid and result.converged
    assert result.adc == pytest.approx(0, abs=1e-10)
    assert result.s0 >= 0


def test_rician_fit_uses_repetition_distribution_instead_of_only_means():
    estimators = _estimators()
    spread = estimators.fit_rician_adc([[1.3, 0.7], [0.8, 0.2]], [0, 1000], sigma=0.3)
    constant = estimators.fit_rician_adc([[1, 1], [0.5, 0.5]], [0, 1000], sigma=0.3)
    assert spread.valid and constant.valid
    assert abs(float(spread.adc - constant.adc)) > 1e-5


def test_rician_fit_excludes_invalid_voxels_and_reports_zero_amplitude():
    magnitude = np.array([[[1, -1, np.nan, np.inf, 0]], [[0.5, 0.5, 0.5, 0.5, 0]]])
    result = _estimators().fit_rician_adc(magnitude, [0, 1000], sigma=0.1)
    np.testing.assert_array_equal(result.valid, [True, False, False, False, False])
    np.testing.assert_array_equal(result.converged, [True, False, False, False, True])
    assert np.all(np.isnan(result.adc[1:]))
    assert np.all(np.isnan(result.s0[1:4]))
    assert result.s0[-1] == 0


@pytest.mark.parametrize("magnitude", [1, [1, 0.5], np.empty((2, 0)), [[1j], [0.5]]])
def test_rician_fit_rejects_invalid_shapes_and_complex_input(magnitude):
    with pytest.raises(ValueError):
        _estimators().fit_rician_adc(magnitude, [0, 1000], sigma=0.1)


@pytest.mark.parametrize("sigma", [0, -1, np.nan, np.inf, [0.1], 1j])
def test_rician_fit_rejects_invalid_sigma(sigma):
    with pytest.raises(ValueError):
        _estimators().fit_rician_adc([[1], [0.5]], [0, 1000], sigma=sigma)


@pytest.mark.parametrize("maxiter", [0, -1, 1.5, True])
def test_rician_fit_rejects_invalid_iteration_limit(maxiter):
    with pytest.raises(ValueError):
        _estimators().fit_rician_adc([[1], [0.5]], [0, 1000], sigma=0.1, maxiter=maxiter)


def test_rician_fit_excludes_iteration_limited_adc_and_preserves_status():
    magnitude = [[1.2, 0.85, 1.1], [0.3, 0.8, 0.55], [0.2, 0.35, 0.1]]
    result = _estimators().fit_rician_adc(magnitude, [0, 500, 1000], sigma=0.2, maxiter=1)
    assert not result.converged
    assert not result.valid
    assert np.isnan(result.adc)
    assert np.isfinite(result.s0)


@pytest.mark.parametrize("sigma", [1e-200, 1e200])
def test_rician_fit_reports_unrepresentable_noise_scale_as_invalid(sigma):
    with np.errstate(all="raise"):
        result = _estimators().fit_rician_adc([[1], [0.5]], [0, 1000], sigma=sigma)
    assert not result.valid
    assert not result.converged
    assert np.isnan(result.adc)


def test_rician_fit_matches_independent_rice_density_maximum():
    from scipy.optimize import minimize_scalar
    from scipy.stats import rice

    sigma = 0.2
    magnitude = np.array([[1.2, 0.8, 1.1, 0.9], [0.55, 0.35, 0.6, 0.4]])
    # For two b-values the independently fitted signal amplitudes determine
    # S0 and ADC, provided the unconstrained amplitudes decrease with b.
    amplitudes = [minimize_scalar(lambda signal: -rice.logpdf(row, signal / sigma, scale=sigma).sum(),
                                 bounds=(0.01, 2), method="bounded", options={"xatol": 1e-12}).x
                  for row in magnitude]
    result = _estimators().fit_rician_adc(magnitude, [0, 1000], sigma=sigma)
    assert result.valid
    assert result.s0 == pytest.approx(amplitudes[0], abs=1e-6)
    assert result.adc == pytest.approx(np.log(amplitudes[0] / amplitudes[1]) / 1000, abs=1e-8)


@pytest.mark.parametrize("b_values", [[0, 1000], [200, 1000]])
def test_rician_fit_excludes_unbounded_two_b_adc(b_values):
    result = _estimators().fit_rician_adc([[1], [0.05]], b_values, sigma=0.2)
    if b_values[0] == 0:
        assert result.converged
    # Nonzero lowest b also sends S0 to infinity along the boundary;
    # numerical convergence may fail independently of identifiability.
    assert not result.valid
    assert np.isnan(result.adc)
    assert np.isfinite(result.s0)


def test_rician_fit_excludes_zero_high_b_mle_using_all_repetitions():
    # A few observations exceed sqrt(2)*sigma, but their mean squared
    # magnitude is below 2*sigma², making the shared amplitude MLE zero.
    magnitude = [[1, 1, 1, 1], [0.4, 0.05, 0.05, 0.05]]
    result = _estimators().fit_rician_adc(magnitude, [0, 1000], sigma=0.2)
    assert result.converged
    assert not result.valid
    assert np.isnan(result.adc)


def test_rician_fit_excludes_unbounded_multi_b_adc():
    result = _estimators().fit_rician_adc([[1], [0.05], [0.05]], [0, 500, 1000], sigma=0.2)
    assert result.converged
    assert not result.valid
    assert np.isnan(result.adc)


def test_rician_fit_keeps_multi_b_adc_identified_by_intermediate_signal():
    result = _estimators().fit_rician_adc([[1], [0.5], [0.05]], [0, 500, 1000], sigma=0.2)
    assert result.converged and result.valid
    assert np.isfinite(result.adc)


def test_rician_fit_does_not_use_mean_magnitude_for_zero_amplitude_criterion():
    # Mean high-b magnitude is below sqrt(2)*sigma, but squared magnitudes
    # have sufficient evidence for a positive high-b amplitude MLE.
    magnitude = [[1, 1, 1, 1], [0.6, 0, 0, 0]]
    result = _estimators().fit_rician_adc(magnitude, [0, 1000], sigma=0.2)
    assert result.converged and result.valid
    assert np.isfinite(result.adc)


@pytest.mark.parametrize("magnitude", [[[0.1], [0.3]], [[1e-5], [0.39999]]])
def test_rician_fit_excludes_exact_zero_amplitude_constrained_optimum(magnitude):
    result = _estimators().fit_rician_adc(magnitude, [0, 1000], sigma=0.2)
    assert result.converged
    assert not result.valid
    assert np.isnan(result.adc)
    assert result.s0 == 0


def test_rician_fit_detects_multi_b_zero_amplitude_prefix_certificate():
    result = _estimators().fit_rician_adc([[0.1], [0.1], [0.35]], [0, 500, 1000], sigma=0.2)
    assert result.converged
    assert not result.valid
    assert np.isnan(result.adc)
    assert result.s0 == 0


def test_rician_fit_does_not_use_pooled_moment_alone_for_zero_amplitude():
    # Pooled squared magnitude is below 2*sigma², but low-b observations
    # support positive amplitude; the maximum instead lies at infinite ADC.
    result = _estimators().fit_rician_adc([[0.35], [0.05], [0.05]], [0, 500, 1000], sigma=0.2)
    assert result.converged
    assert not result.valid
    assert np.isnan(result.adc)
    assert result.s0 > 0


def test_rician_fit_excludes_conditional_null_amplitude_without_prefix_certificate():
    # The middle-b positive moment creates a positive prefix sum, so the
    # sufficient global prefix certificate does not apply. At the returned
    # D, however, the exact weighted score still makes S0=0 optimal.
    result = _estimators().fit_rician_adc([[0.1], [0.4], [0.05]], [0, 500, 1000], sigma=0.2)
    assert result.converged
    assert not result.valid
    assert np.isnan(result.adc)
