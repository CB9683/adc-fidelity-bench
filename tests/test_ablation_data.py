"""Known-answer paired data tests for the focal-preservation experiment."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


def small_config():
    config = json.loads((Path(__file__).parents[1] / 'configs/focal_ablation.json').read_text())
    config['simulation']['anatomy_ids'] = ['abl-a', 'abl-b', 'abl-c', 'abl-d']
    config['simulation']['split_counts'] = dict.fromkeys(('train', 'validation', 'calibration', 'test'), 1)
    config['simulation']['snr'] = [20]
    config['simulation']['budgets'] = [1]
    config['simulation']['focal_radius_mm'] = [0.5]
    return config


def test_new_ablation_data_module_is_available():
    from adc_fidelity_bench.learning.ablation_data import ablation_splits, iter_ablation_cases
    assert callable(ablation_splits) and callable(iter_ablation_cases)


def test_frozen_config_defines_fresh_disjoint_cohort_and_all_prescribed_controls():
    from adc_fidelity_bench.learning.ablation_data import ablation_splits
    config = json.loads((Path(__file__).parents[1] / 'configs/focal_ablation.json').read_text())
    splits = ablation_splits(config)
    assert {key: len(ids) for key, ids in splits.items()} == {
        'train': 64, 'validation': 16, 'calibration': 32, 'test': 32}
    assert len(set(identifier for ids in splits.values() for identifier in ids)) == 144
    assert all(identifier.startswith('abl-') for ids in splits.values() for identifier in ids)
    assert config['simulation']['delta_adc'] == [-0.0003, -0.00015, 0, 0.00015, 0.0003]
    assert config['evaluation']['noise_realizations'] == 3


def test_varied_parameters_locations_noise_and_null_signals_are_shared_across_contrasts():
    from adc_fidelity_bench.learning.ablation_data import iter_ablation_cases
    from adc_fidelity_bench.learning.data import LearningCase
    config = small_config()
    cases = list(iter_ablation_cases(config, 'train'))
    assert len(cases) == 10  # five contrasts and two noise draws, one geometry
    again = list(iter_ablation_cases(deepcopy(config), 'train'))
    for case, repeat in zip(cases, again):
        assert isinstance(case, LearningCase) and case.metadata == repeat.metadata
        np.testing.assert_array_equal(case.healthy_magnitude, repeat.healthy_magnitude)
        np.testing.assert_array_equal(case.perturbed_magnitude, repeat.perturbed_magnitude)
        assert case.metadata['regime'] == 'varied'
        assert case.metadata['noise_seed_scope'] == 'shared_across_delta'
        assert case.healthy_magnitude.shape == (2, 1, 8, 8)
        for parameter in ('adc_inner', 'adc_outer', 's0_inner', 's0_outer'):
            low, high = config['ablation'][f'{parameter}_range']
            assert low <= case.metadata[parameter] <= high
    for realization in range(2):
        variants = [case for case in cases if case.metadata['noise_realization'] == realization]
        shared_names = ('anatomy_seed', 'tissue_seed', 'location_seed', 'focal_center_y_mm',
                        'focal_center_x_mm', 'adc_inner', 'adc_outer', 's0_inner', 's0_outer',
                        'healthy_noise_seed', 'perturbed_noise_seed')
        assert all(len({case.metadata[name] for case in variants}) == 1 for name in shared_names)
        for case in variants:
            np.testing.assert_array_equal(case.healthy_magnitude, variants[0].healthy_magnitude)
            np.testing.assert_array_equal(case.focal_roi, variants[0].focal_roi)
        null = next(case for case in variants if case.metadata['delta_adc'] == 0)
        assert null.metadata['case_kind'] == 'null'
        np.testing.assert_array_equal(null.reference, null.healthy_reference)
        assert null.metadata['healthy_noise_seed'] != null.metadata['perturbed_noise_seed']
        assert not np.array_equal(null.healthy_magnitude, null.perturbed_magnitude)
        assert all(case.metadata['case_kind'] == 'change' for case in variants if case.metadata['delta_adc'] != 0)


def test_original_regime_exactly_matches_existing_noiseless_phantom_reference():
    from adc_fidelity_bench.learning.ablation_data import iter_ablation_cases
    from adc_fidelity_bench.baselines import fit_log_linear_adc
    from adc_fidelity_bench.simulation.phantoms import make_toy_phantom
    from adc_fidelity_bench.simulation.partial_volume import block_average
    from adc_fidelity_bench.simulation.signals import monoexponential_signal
    config = small_config()
    case = next(iter_ablation_cases(config, 'test', regimes=['original'], focal_shapes=['ellipse'], noise_realizations=1))
    meta, simulation = case.metadata, config['simulation']
    phantom = make_toy_phantom(anatomy_seed=meta['anatomy_seed'], shape=simulation['fine_shape'],
                              voxel_spacing_mm=simulation['fine_spacing_mm'],
                              focal_radius_mm=meta['focal_radius_mm'], delta_adc=meta['delta_adc'],
                              focal_shape=meta['focal_shape'])
    expected = fit_log_linear_adc(block_average(monoexponential_signal(
        simulation['b_values'], phantom.adc_perturbed, phantom.s0), 2), simulation['b_values']).adc
    expected_healthy = fit_log_linear_adc(block_average(monoexponential_signal(
        simulation['b_values'], phantom.adc_unperturbed, phantom.s0), 2), simulation['b_values']).adc
    np.testing.assert_array_equal(case.reference, expected)
    np.testing.assert_array_equal(case.healthy_reference, expected_healthy)
    np.testing.assert_array_equal(case.focal_roi, block_average(phantom.focal_mask, 2) > 0)
    assert (meta['adc_inner'], meta['adc_outer'], meta['s0_inner'], meta['s0_outer']) == (0.0009, 0.0016, 0.8, 1.0)


def test_varied_reference_and_noise_are_constructed_from_physical_signals_after_resampling():
    from adc_fidelity_bench.learning.ablation_data import iter_ablation_cases
    from adc_fidelity_bench.simulation.phantoms import make_toy_phantom
    from adc_fidelity_bench.simulation.partial_volume import block_average
    from adc_fidelity_bench.simulation.noise import add_complex_noise
    config = small_config()
    case = next(iter_ablation_cases(config, 'train'))
    meta, sim = case.metadata, config['simulation']
    base = make_toy_phantom(anatomy_seed=meta['anatomy_seed'], shape=sim['fine_shape'],
                           voxel_spacing_mm=sim['fine_spacing_mm'], focal_radius_mm=0.5, delta_adc=0)
    inner = base.adc_unperturbed == 0.0009
    adc = np.where(inner, meta['adc_inner'], np.where(base.tissue_mask, meta['adc_outer'], 0.0))
    s0 = np.where(inner, meta['s0_inner'], np.where(base.tissue_mask, meta['s0_outer'], 0.0))
    y, x = np.indices(sim['fine_shape'], dtype=float)
    angle = meta['focal_angle_radians']
    xrot = (x - meta['focal_center_x_px']) * np.cos(angle) + (y - meta['focal_center_y_px']) * np.sin(angle)
    yrot = -(x - meta['focal_center_x_px']) * np.sin(angle) + (y - meta['focal_center_y_px']) * np.cos(angle)
    radius = meta['focal_radius_mm'] / sim['fine_spacing_mm']
    focal = (xrot / radius) ** 2 + (yrot / radius) ** 2 <= 1
    assert focal.any() and not np.any(focal & ~base.tissue_mask)
    adc[focal] += meta['delta_adc']
    low = s0.reshape(8, 2, 8, 2).mean(axis=(1, 3))
    high = (s0 * np.exp(-1000 * adc)).reshape(8, 2, 8, 2).mean(axis=(1, 3))
    np.testing.assert_allclose(case.reference[case.tissue_roi], np.log(low[case.tissue_roi] / high[case.tissue_roi]) / 1000,
                               atol=1e-15)
    signals = np.stack([low, high])
    expected_magnitude = np.abs(add_complex_noise(signals, repetitions=1, sigma=meta['sigma'],
                                                rng=np.random.default_rng(meta['perturbed_noise_seed'])))
    np.testing.assert_allclose(case.perturbed_magnitude, expected_magnitude, atol=1e-15)
    np.testing.assert_array_equal(case.focal_roi, block_average(focal, 2) > 0)
    assert np.isnan(case.reference[~case.tissue_roi]).all()


def test_evaluation_roles_regimes_shapes_and_pairing_are_explicit_and_do_not_mutate_config():
    from adc_fidelity_bench.learning.ablation_data import ablation_splits, iter_ablation_cases
    config = small_config()
    before = deepcopy(config)
    cases = list(iter_ablation_cases(config, 'test'))
    assert len(cases) == 60  # two regimes * two shapes * five contrasts * three draws
    assert {case.metadata['regime'] for case in cases} == {'original', 'varied'}
    assert {case.metadata['focal_shape'] for case in cases} == {'circle', 'ellipse'}
    assert {case.metadata['anatomy_id'] for case in cases} == set(ablation_splits(config)['test'])
    common = next(case for case in iter_ablation_cases(config, 'test', paired_noise='common') if case.metadata['case_kind'] == 'null')
    np.testing.assert_array_equal(common.healthy_magnitude, common.perturbed_magnitude)
    assert config == before


def test_tissue_parameters_ignore_snr_noise_size_shape_and_locations_ignore_snr_noise():
    from adc_fidelity_bench.learning.ablation_data import iter_ablation_cases
    config = small_config()
    config['simulation']['snr'] = [5, 20]
    config['simulation']['focal_radius_mm'] = [0.5, 1.5]
    cases = list(iter_ablation_cases(config, 'test', regimes=['varied'], noise_realizations=2))
    for name in ('tissue_seed', 'adc_inner', 'adc_outer', 's0_inner', 's0_outer'):
        assert len({case.metadata[name] for case in cases}) == 1
    for radius in (0.5, 1.5):
        for shape in ('circle', 'ellipse'):
            group = [case for case in cases if case.metadata['focal_radius_mm'] == radius and case.metadata['focal_shape'] == shape]
            for name in ('location_seed', 'focal_center_y_px', 'focal_center_x_px'):
                assert len({case.metadata[name] for case in group}) == 1
            for case in group:
                np.testing.assert_array_equal(case.focal_roi, group[0].focal_roi)


@pytest.mark.parametrize('kwargs', [
    {'split': 'train', 'regimes': ['original']}, {'split': 'validation', 'focal_shapes': ['ellipse']},
    {'split': 'missing'}, {'split': 'calibration'}, {'split': 'test', 'regimes': ['unknown']},
    {'split': 'test', 'focal_shapes': ['square']}, {'split': 'test', 'paired_noise': 'unknown'},
    {'split': 'test', 'noise_realizations': True}, {'split': 'test', 'noise_realizations': 0},
])
def test_invalid_case_requests_are_rejected(kwargs):
    from adc_fidelity_bench.learning.ablation_data import iter_ablation_cases
    with pytest.raises(ValueError):
        list(iter_ablation_cases(small_config(), **kwargs))


@pytest.mark.parametrize('edit', [
    lambda cfg: cfg.update(unknown=True),
    lambda cfg: cfg['simulation']['anatomy_ids'].__setitem__(0, 'toy-000'),
    lambda cfg: cfg['simulation']['anatomy_ids'].__setitem__(0, 'cnn-000'),
    lambda cfg: cfg['simulation'].update(focal_radius_mm=[2.0]),
    lambda cfg: cfg['simulation'].update(delta_adc=[-0.002, 0, 0.0003]),
    lambda cfg: cfg['simulation'].update(delta_adc=[-0.0003, 0.0003]),
    lambda cfg: cfg['ablation'].update(adc_inner_range=[0.0001, 0.0002]),
    lambda cfg: cfg['ablation'].update(adc_outer_range=[0.0019, 0.0013]),
    lambda cfg: cfg['ablation'].update(s0_inner_range=[0, 0.95]),
    lambda cfg: cfg['ablation'].update(s0_outer_range=[0.85, float('nan')]),
    lambda cfg: cfg['ablation'].update(training_regimes=['original']),
    lambda cfg: cfg['simulation'].update(focal_shapes=['ellipse']),
    lambda cfg: cfg['ablation'].update(evaluation_regimes=['invalid']),
    lambda cfg: cfg['ablation'].update(change_loss_weight=-1),
    lambda cfg: cfg['ablation'].update(selection_change_weight=True),
    lambda cfg: cfg['ablation'].update(null_threshold_adc=0),
])
def test_configuration_fails_early_for_invalid_controls_or_geometry(edit):
    from adc_fidelity_bench.learning.ablation_data import ablation_splits
    config = small_config()
    edit(config)
    with pytest.raises(ValueError):
        ablation_splits(config)


def test_ablation_data_import_does_not_load_optional_torch():
    result = subprocess.run([sys.executable, '-c',
                             'import sys; import adc_fidelity_bench.learning.ablation_data; assert "torch" not in sys.modules'],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
