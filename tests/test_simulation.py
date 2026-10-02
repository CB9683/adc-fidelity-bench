import numpy as np
import pytest


def test_monoexponential_units_and_spatial_axes():
    from adc_fidelity_bench.simulation.signals import monoexponential_signal

    adc = np.array([[0.001, 0.002], [0.0, 0.0005]])
    s0 = np.array([[2.0, 3.0], [1.0, 0.5]])
    signal = monoexponential_signal([0.0, 1000.0], adc, s0)
    assert signal.shape == (2, 2, 2)
    np.testing.assert_allclose(signal[0], s0)
    np.testing.assert_allclose(np.log(signal[0] / signal[1]) / 1000.0, adc)


@pytest.mark.parametrize('b_values,adc,s0', [
    ([], 0.001, 1.0), ([0, 0], 0.001, 1.0), ([1000, 0], 0.001, 1.0),
    ([0, -100], 0.001, 1.0), ([0, np.nan], 0.001, 1.0),
    ([0, 1000], -0.001, 1.0), ([0, 1000], np.inf, 1.0),
    ([0, 1000], 0.001, -1.0), ([0, 1000], 0.001, np.nan),
])
def test_signal_generation_rejects_undefined_or_unphysical_parameters(b_values, adc, s0):
    from adc_fidelity_bench.simulation.signals import monoexponential_signal

    with pytest.raises(ValueError):
        monoexponential_signal(b_values, adc, s0)


def test_tissue_mixing_is_performed_on_signals():
    from adc_fidelity_bench.simulation.signals import mix_tissue_signals

    fractions = np.array([[[0.5]], [[0.5]]])
    signal = mix_tissue_signals(fractions, [0.0, 0.002], [1.0, 1.0], [0, 1000])
    np.testing.assert_allclose(signal[:, 0, 0], [1.0, (1.0 + np.exp(-2.0)) / 2.0])
    reference = np.log(signal[0, 0, 0] / signal[1, 0, 0]) / 1000.0
    assert abs(reference - 0.001) > 0.0004


@pytest.mark.parametrize('fractions', [
    np.array([[[0.6]], [[0.6]]]), np.array([[[-0.1]], [[1.1]]]),
    np.array([[[np.nan]], [[1.0]]]),
])
def test_tissue_fractions_are_validated(fractions):
    from adc_fidelity_bench.simulation.signals import mix_tissue_signals

    with pytest.raises(ValueError):
        mix_tissue_signals(fractions, [0.0, 0.002], [1.0, 1.0], [0, 1000])


def test_spatial_partial_volume_keeps_b_value_axis_and_differs_from_blurred_adc():
    from adc_fidelity_bench.simulation.partial_volume import block_average
    from adc_fidelity_bench.simulation.signals import monoexponential_signal

    adc = np.array([[0.0, 0.002], [0.0, 0.002]])
    signal = monoexponential_signal([0, 1000], adc)
    coarse = block_average(signal, 2)
    assert coarse.shape == (2, 1, 1)
    np.testing.assert_allclose(coarse[0], 1.0)
    fitted_reference = np.log(coarse[0] / coarse[1]) / 1000.0
    assert not np.allclose(fitted_reference, block_average(adc, 2), atol=1e-5)


def test_partial_volume_preserves_complex_signal_phase():
    from adc_fidelity_bench.simulation.partial_volume import block_average

    signal = np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]])
    np.testing.assert_array_equal(block_average(signal, 2), [[4 + 5j]])


@pytest.mark.parametrize('data,factor', [
    (np.ones((3, 4)), 2), (np.ones((4, 4)), 0),
    (np.ones((4, 4)), 1.5), (np.ones((4, 4)), True),
    (np.array([[np.nan]]), 1), (np.ones(4), 1),
])
def test_partial_volume_rejects_ambiguous_sampling(data, factor):
    from adc_fidelity_bench.simulation.partial_volume import block_average

    with pytest.raises(ValueError):
        block_average(data, factor)


def test_complex_noise_repetitions_and_zero_noise_limit():
    from adc_fidelity_bench.simulation.noise import add_complex_noise

    signal = np.arange(8.0).reshape(2, 2, 2)
    noisy = add_complex_noise(signal, repetitions=3, sigma=0, rng=np.random.default_rng(2))
    assert noisy.shape == (2, 3, 2, 2)
    assert np.iscomplexobj(noisy)
    np.testing.assert_array_equal(noisy.real, np.repeat(signal[:, None], 3, axis=1))
    np.testing.assert_array_equal(noisy.imag, 0.0)


def test_noise_is_seeded_complex_gaussian_before_magnitude():
    from adc_fidelity_bench.simulation.noise import add_complex_noise

    first = add_complex_noise(np.array([0.0]), repetitions=100000, sigma=2.0,
                              rng=np.random.default_rng(2026))
    second = add_complex_noise(np.array([0.0]), repetitions=100000, sigma=2.0,
                               rng=np.random.default_rng(2026))
    np.testing.assert_array_equal(first, second)
    assert abs(first.real.mean()) < 0.03
    assert abs(first.imag.mean()) < 0.03
    assert first.real.std() == pytest.approx(2.0, abs=0.03)
    assert first.imag.std() == pytest.approx(2.0, abs=0.03)
    assert np.abs(first).mean() == pytest.approx(2.0 * np.sqrt(np.pi / 2), abs=0.03)


@pytest.mark.parametrize('repetitions,sigma', [(0, 1), (True, 1), (1.5, 1), (1, -1), (1, np.nan)])
def test_noise_parameters_are_explicit(repetitions, sigma):
    from adc_fidelity_bench.simulation.noise import add_complex_noise

    with pytest.raises(ValueError):
        add_complex_noise(np.ones((2, 2, 2)), repetitions=repetitions, sigma=sigma,
                          rng=np.random.default_rng(1))


def test_perturbations_share_anatomy_but_have_signed_known_changes():
    from adc_fidelity_bench.simulation.phantoms import make_toy_phantom

    common = dict(anatomy_seed=42, shape=(32, 32), voxel_spacing_mm=0.5,
                  focal_radius_mm=1.0, focal_shape='circle')
    increased = make_toy_phantom(**common, delta_adc=0.0003)
    decreased = make_toy_phantom(**common, delta_adc=-0.0003)
    np.testing.assert_array_equal(increased.adc_unperturbed, decreased.adc_unperturbed)
    np.testing.assert_array_equal(increased.s0, decreased.s0)
    assert increased.focal_mask.any()
    assert np.all(increased.tissue_mask[increased.focal_mask])
    np.testing.assert_allclose((increased.adc_perturbed - increased.adc_unperturbed)[increased.focal_mask], 0.0003)
    np.testing.assert_allclose((decreased.adc_perturbed - decreased.adc_unperturbed)[decreased.focal_mask], -0.0003)
    np.testing.assert_array_equal(increased.adc_perturbed[~increased.focal_mask],
                                  increased.adc_unperturbed[~increased.focal_mask])
    other = make_toy_phantom(**{**common, 'anatomy_seed': 43}, delta_adc=0.0003)
    assert not np.array_equal(other.adc_unperturbed, increased.adc_unperturbed)


def test_negative_perturbed_diffusivity_is_rejected():
    from adc_fidelity_bench.simulation.phantoms import make_toy_phantom

    with pytest.raises(ValueError):
        make_toy_phantom(anatomy_seed=42, shape=(32, 32), voxel_spacing_mm=0.5,
                         focal_radius_mm=1.0, delta_adc=-0.01, focal_shape='circle')
