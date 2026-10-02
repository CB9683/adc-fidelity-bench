import csv
import json

import numpy as np
import pytest

from adc_fidelity_bench.pipeline import run_benchmark, validate_config


def tiny_config():
    return {
        'schema_version': 1, 'seed': 17,
        'anatomy_ids': ['toy-a', 'toy-b'],
        'split_counts': {'train': 1, 'validation': 0, 'calibration': 0, 'test': 1},
        'evaluation_splits': ['test'], 'fine_shape': [8, 8],
        'fine_spacing_mm': 0.5, 'partial_volume_factor': 2,
        'b_values': [0, 1000], 'budgets': [1, 2], 'snr': [20],
        'focal_radius_mm': [0.5], 'delta_adc': [-0.0003, 0.0003],
        'focal_shapes': ['circle'], 'noise_realizations': 1,
        'paired_noise': 'common', 'methods': ['average_log', 'rician_mle'],
        'rician_maxiter': 200, 'recovery_epsilon': 1e-8,
    }


def read_rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def test_executable_cases_are_deterministic_and_inherit_anatomy_split(tmp_path):
    first, second = tmp_path / 'first', tmp_path / 'second'
    run_benchmark(tiny_config(), first)
    run_benchmark(tiny_config(), second)
    assert (first / 'cases.csv').read_bytes() == (second / 'cases.csv').read_bytes()
    rows = read_rows(first / 'cases.csv')
    assert len(rows) == 8  # anatomy * contrasts * budgets * methods
    manifest = json.loads((first / 'manifest.json').read_text())
    assert set(row['anatomy_id'] for row in rows) == set(manifest['splits']['test'])
    assert not set(row['anatomy_id'] for row in rows) & set(manifest['splits']['train'])
    for row in rows:
        assert row['split'] == 'test'
        assert int(row['total_acquisitions']) == 2 * int(row['budget_per_b'])
        assert int(row['n_requested']) == int(row['n_valid']) + int(row['n_invalid'])
        assert int(row['focal_n_requested']) == int(row['focal_n_valid']) + int(row['focal_n_invalid'])
        assert int(row['focal_adc_n_requested']) == int(row['focal_n_requested'])
        assert int(row['focal_adc_n_requested']) + int(row['nonfocal_adc_n_requested']) == int(row['n_requested'])
        assert int(row['nonfocal_adc_n_requested']) == int(row['nonfocal_adc_n_valid']) + int(row['nonfocal_adc_n_invalid'])
        # Voxelwise fits with shared noise create no change outside focal footprints.
        assert float(row['nonfocal_change_rmse']) == pytest.approx(0, abs=1e-15)
        assert int(row['healthy_noise_seed']) == int(row['perturbed_noise_seed'])
    assert manifest['noise_pool_repetitions'] == 2
    assert manifest['sigma_definition'] == 'independent complex-component standard deviation'
    assert manifest['reference_definition'] == 'log-linear ADC fitted to noiseless block-averaged signals'
    assert (first / 'config.json').is_file()
    assert (first / 'summary.csv').is_file()
    assert (first / 'anatomy_summary.csv').is_file()
    assert (first / 'error_by_budget.png').is_file()
    assert (first / 'focal_recovery.png').is_file()
    # Independent two-b formula checks the pipeline's signal-derived reference.
    example = np.load(first / 'example.npz')
    s0 = example['fine_s0'].reshape(4, 2, 4, 2)
    adc = example['fine_adc_perturbed'].reshape(4, 2, 4, 2)
    low = s0.mean(axis=(1, 3))
    high = (s0 * np.exp(-1000 * adc)).mean(axis=(1, 3))
    roi = low > 0
    expected = np.log(low[roi] / high[roi]) / 1000
    np.testing.assert_allclose(example['reference'][roi], expected, atol=1e-15)
    assert np.any((example['focal_fraction'] > 0) & (example['focal_fraction'] < 1))


def test_repetitions_are_nested_prefixes_and_estimators_receive_only_observations(tmp_path, monkeypatch):
    from adc_fidelity_bench import pipeline
    original = pipeline.fit_rician_adc
    calls = []
    def spy(magnitude, b_values, *, sigma, maxiter):
        calls.append((magnitude.copy(), sigma))
        return original(magnitude, b_values, sigma=sigma, maxiter=maxiter)
    monkeypatch.setattr(pipeline, 'fit_rician_adc', spy)
    config = tiny_config()
    config['delta_adc'] = [0.0003]
    config['methods'] = ['rician_mle']
    run_benchmark(config, tmp_path / 'nested')
    assert len(calls) == 4  # healthy, perturbed at each budget
    for small, large in zip(calls[:2], calls[2:]):
        np.testing.assert_array_equal(small[0], large[0][:, :1])
        assert small[1] == large[1] == 0.05
        assert np.all(small[0] >= 0)


def test_independent_paired_noise_is_recorded(tmp_path):
    config = tiny_config()
    config['paired_noise'] = 'independent'
    config['methods'] = ['average_log']
    run_benchmark(config, tmp_path / 'independent')
    for row in read_rows(tmp_path / 'independent' / 'cases.csv'):
        assert row['paired_noise'] == 'independent'
        assert row['healthy_noise_seed'] != row['perturbed_noise_seed']


@pytest.mark.parametrize('key,value', [
    ('seed', None), ('seed', True), ('budgets', [0]), ('budgets', [2, 1]),
    ('snr', [0]), ('snr', [1e-320]), ('b_values', [0, 0]), ('partial_volume_factor', 3),
    ('methods', ['oracle']), ('noise_realizations', 0), ('paired_noise', 'unspecified'),
    ('evaluation_splits', ['missing']), ('recovery_epsilon', 0),
    ('focal_radius_mm', [9]), ('delta_adc', [-1]), ('fine_shape', [8, 7]),
])
def test_invalid_configuration_creates_no_output(tmp_path, key, value):
    config = tiny_config()
    config[key] = value
    with pytest.raises(ValueError):
        run_benchmark(config, tmp_path / 'invalid')
    assert not (tmp_path / 'invalid').exists()


def test_unknown_configuration_field_is_rejected():
    config = tiny_config()
    config['budegt'] = 2
    with pytest.raises(ValueError):
        validate_config(config)


def test_existing_results_are_preserved(tmp_path):
    output = tmp_path / 'existing'
    output.mkdir()
    (output / 'important.csv').write_text('preserve me')
    with pytest.raises(FileExistsError):
        run_benchmark(tiny_config(), output)
    assert (output / 'important.csv').read_text() == 'preserve me'
    assert list(output.iterdir()) == [output / 'important.csv']


def test_anatomy_aggregation_does_not_treat_noise_variants_as_anatomies(tmp_path):
    config = tiny_config()
    config['noise_realizations'] = 2
    config['methods'] = ['average_log']
    run_benchmark(config, tmp_path / 'aggregation')
    rows = read_rows(tmp_path / 'aggregation' / 'summary.csv')
    assert len(rows) == 4
    assert all(int(row['n_anatomies']) == 1 and int(row['n_cases']) == 2 for row in rows)
