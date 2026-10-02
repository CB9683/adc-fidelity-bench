"""Learning examples preserve anatomy groups and acquisition-space references."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from adc_fidelity_bench.learning.data import iter_learning_cases, learning_splits
from adc_fidelity_bench.pipeline import stable_seed
from adc_fidelity_bench.simulation.noise import add_complex_noise
from adc_fidelity_bench.simulation.partial_volume import block_average
from adc_fidelity_bench.simulation.phantoms import make_toy_phantom
from adc_fidelity_bench.simulation.signals import monoexponential_signal


def small_config():
    config = json.loads((Path(__file__).parents[1] / 'configs/first_cnn.json').read_text())
    simulation = config['simulation']
    simulation['anatomy_ids'] = ['cnn-a', 'cnn-b', 'cnn-c', 'cnn-d']
    simulation['split_counts'] = dict.fromkeys(('train', 'validation', 'calibration', 'test'), 1)
    simulation['seed'] = 17
    simulation['budgets'] = [2]
    simulation['snr'] = [20]
    simulation['focal_radius_mm'] = [0.5]
    simulation['delta_adc'] = [-0.0003, 0.0003]
    return config


def test_learning_roles_keep_whole_geometries_disjoint_and_exclude_old_pilot_ids():
    config = small_config()
    splits = learning_splits(config)
    assert list(splits) == ['train', 'validation', 'calibration', 'test']
    assert all(len(ids) == 1 for ids in splits.values())
    assert len(set(identifier for ids in splits.values() for identifier in ids)) == 4
    config['simulation']['anatomy_ids'][0] = 'toy-000'
    with pytest.raises(ValueError, match='pilot|toy-'):
        learning_splits(config)


def test_examples_are_deterministic_and_use_split_specific_noise_counts():
    config = small_config()
    splits = learning_splits(config)
    for split, expected in [('train', 8), ('validation', 4), ('test', 2)]:
        first = list(iter_learning_cases(config, split))
        second = list(iter_learning_cases(deepcopy(config), split))
        assert len(first) == expected  # two contrasts, independently generated noise draws
        assert {case.metadata['anatomy_id'] for case in first} == set(splits[split])
        assert {case.metadata['split'] for case in first} == {split}
        assert {case.metadata['noise_realization'] for case in first} == set(range(expected // 2))
        for a, b in zip(first, second):
            assert a.metadata == b.metadata
            np.testing.assert_array_equal(a.healthy_magnitude, b.healthy_magnitude)
            np.testing.assert_array_equal(a.perturbed_magnitude, b.perturbed_magnitude)
            assert a.healthy_magnitude.shape == a.perturbed_magnitude.shape == (2, 2, 8, 8)
            assert a.reference.shape == a.healthy_reference.shape == (8, 8)
            assert a.tissue_roi.dtype == a.focal_roi.dtype == np.bool_
            assert a.focal_roi.any() and np.all(a.tissue_roi[a.focal_roi])
            assert a.metadata['budget_per_b'] == 2 and a.metadata['total_acquisitions'] == 4
            assert a.metadata['sigma'] == 0.05 and a.metadata['paired_noise'] == 'independent'


def test_reference_is_fit_after_signal_resampling_and_noise_follows_resampling():
    config = small_config()
    case = next(iter_learning_cases(config, 'test'))
    simulation = config['simulation']
    metadata = case.metadata
    phantom = make_toy_phantom(
        anatomy_seed=stable_seed(simulation['seed'], 'anatomy', metadata['anatomy_id']),
        shape=simulation['fine_shape'], voxel_spacing_mm=simulation['fine_spacing_mm'],
        focal_radius_mm=metadata['focal_radius_mm'], delta_adc=metadata['delta_adc'],
        focal_shape=metadata['focal_shape'])
    fine_s0 = phantom.s0.reshape(8, 2, 8, 2)
    fine_adc = phantom.adc_perturbed.reshape(8, 2, 8, 2)
    low = fine_s0.mean(axis=(1, 3))
    high = (fine_s0 * np.exp(-1000 * fine_adc)).mean(axis=(1, 3))
    expected_reference = np.log(low[case.tissue_roi] / high[case.tissue_roi]) / 1000
    np.testing.assert_allclose(case.reference[case.tissue_roi], expected_reference, atol=1e-15)
    coarse_signal = block_average(monoexponential_signal(simulation['b_values'],
                                  phantom.adc_perturbed, phantom.s0), 2)
    expected_magnitude = np.abs(add_complex_noise(
        coarse_signal, repetitions=2, sigma=metadata['sigma'],
        rng=np.random.default_rng(metadata['perturbed_noise_seed'])))
    np.testing.assert_array_equal(case.perturbed_magnitude, expected_magnitude)
    assert np.isnan(case.reference[~case.tissue_roi]).all()
    expected_focal_fraction = block_average(phantom.focal_mask, 2)
    np.testing.assert_array_equal(case.focal_roi, expected_focal_fraction > 0)
    assert metadata['focal_area_mm2'] == pytest.approx(phantom.focal_mask.sum() * 0.25)
    assert metadata['focal_coarse_fraction_sum'] == pytest.approx(expected_focal_fraction.sum())


def test_common_noise_pairing_is_an_explicit_control_with_identical_nonfocal_inputs():
    config = small_config()
    common = next(iter_learning_cases(config, 'test', paired_noise='common'))
    independent = next(iter_learning_cases(config, 'test', paired_noise='independent'))
    assert common.metadata['healthy_noise_seed'] == common.metadata['perturbed_noise_seed']
    assert independent.metadata['healthy_noise_seed'] != independent.metadata['perturbed_noise_seed']
    np.testing.assert_array_equal(common.healthy_magnitude, independent.healthy_magnitude)
    nonfocal = common.tissue_roi & ~common.focal_roi
    np.testing.assert_array_equal(common.healthy_magnitude[..., nonfocal],
                                  common.perturbed_magnitude[..., nonfocal])
    assert not np.array_equal(independent.healthy_magnitude[..., nonfocal],
                              independent.perturbed_magnitude[..., nonfocal])


def test_shape_and_noise_overrides_are_explicit_without_changing_geometry_assignments():
    config = small_config()
    before = deepcopy(config)
    circles = list(iter_learning_cases(config, 'test', noise_realizations=1))
    ellipses = list(iter_learning_cases(config, 'test', focal_shapes=['ellipse'], noise_realizations=2))
    assert len(circles) == 2 and len(ellipses) == 4
    assert {case.metadata['focal_shape'] for case in ellipses} == {'ellipse'}
    assert {case.metadata['anatomy_id'] for case in circles} == {case.metadata['anatomy_id'] for case in ellipses}
    assert {case.metadata['anatomy_seed'] for case in circles} == {case.metadata['anatomy_seed'] for case in ellipses}
    assert config == before


def test_calibration_generation_requires_explicit_request():
    config = small_config()
    with pytest.raises(ValueError, match='calibration.*explicit|explicit.*calibration'):
        list(iter_learning_cases(config, 'calibration'))
    cases = list(iter_learning_cases(config, 'calibration', noise_realizations=1))
    assert len(cases) == 2 and {case.metadata['split'] for case in cases} == {'calibration'}


def test_registered_evaluation_geometry_is_validated_before_training():
    config = small_config()
    config['simulation']['focal_radius_mm'] = [2.0]
    before = deepcopy(config)
    # This circle fits the training phantoms, while its equal-area ellipse
    # extends outside tissue in at least one preassigned evaluation geometry.
    with pytest.raises(ValueError, match='focal region'):
        learning_splits(config)
    assert config == before


@pytest.mark.parametrize('edit', [
    lambda config: config.update(unknown=1),
    lambda config: config.update(schema_version=2),
    lambda config: config['simulation'].update(b_values=[0, 800]),
    lambda config: config['simulation'].update(budgets=[1, 2]),
    lambda config: config['simulation']['split_counts'].update(train=0, test=2),
    lambda config: config['training'].update(train_noise_realizations=0),
    lambda config: config['training'].update(validation_noise_realizations=True),
    lambda config: config['evaluation'].update(noise_realizations=0),
    lambda config: config['evaluation'].update(focal_shapes=['square']),
    lambda config: config['evaluation'].update(paired_noise=['unknown']),
])
def test_incompatible_learning_configuration_is_rejected(edit):
    config = small_config()
    edit(config)
    with pytest.raises(ValueError):
        learning_splits(config)


@pytest.mark.parametrize('kwargs', [
    {'split': 'missing'}, {'split': 'test', 'noise_realizations': 0},
    {'split': 'test', 'noise_realizations': True}, {'split': 'test', 'focal_shapes': []},
    {'split': 'test', 'focal_shapes': ['square']}, {'split': 'test', 'paired_noise': 'unknown'},
])
def test_bad_case_request_is_rejected(kwargs):
    with pytest.raises(ValueError):
        list(iter_learning_cases(small_config(), **kwargs))


def test_learning_data_import_does_not_load_optional_torch():
    result = subprocess.run(
        [sys.executable, '-c', 'import sys; import adc_fidelity_bench.learning.data; assert "torch" not in sys.modules'],
        capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
