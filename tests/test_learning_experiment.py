import csv
import json
from pathlib import Path

import numpy as np
import pytest

def tiny_config():
    root = Path(__file__).resolve().parents[1]
    c = json.loads((root / 'configs/first_cnn.json').read_text())
    c['simulation'].update(anatomy_ids=[f'smoke-{i}' for i in range(8)],
                           split_counts=dict(train=2, validation=2, calibration=2, test=2),
                           fine_shape=[8, 8], budgets=[2], snr=[20],
                           focal_radius_mm=[0.5], delta_adc=[-0.0003, 0.0003])
    c['training'].update(seeds=[11], epochs=3, batch_size=8, learning_rate=0.01,
                         train_noise_realizations=1, validation_noise_realizations=1)
    c['evaluation'].update(focal_shapes=['circle'], paired_noise=['common'], noise_realizations=1)
    return c


def experiment_module():
    import adc_fidelity_bench.learning as learning
    assert (Path(learning.__file__).parent / 'experiment.py').exists(), 'CNN experiment is not implemented'
    pytest.importorskip('torch')
    from adc_fidelity_bench.learning import experiment
    return experiment


def test_training_selects_validation_checkpoint_and_reload_reproduces_prediction(tmp_path):
    e = experiment_module()
    c = tiny_config()
    output = tmp_path / 'run'
    result = e.run_cnn_experiment(c, output, evaluate=False)
    assert result['manifest']['status'] == 'trained'
    history = list(csv.DictReader((output / 'training_history.csv').open()))
    assert len(history) == 3
    assert all(float(r['train_loss']) >= 0 for r in history)
    selected = result['manifest']['checkpoints'][0]
    assert selected['validation_loss'] == min(float(r['validation_loss']) for r in history)
    assert selected['validation_loss'] < selected['initial_validation_loss']
    from adc_fidelity_bench.learning.model import predict_adc
    model = e.load_model_checkpoint(output / selected['filename'])
    sample = np.ones((2, 2, 4, 4), dtype=float)
    a = predict_adc(model, sample)
    b = predict_adc(e.load_model_checkpoint(output / selected['filename']), sample)
    np.testing.assert_array_equal(a, b)
    assert selected['parameter_count'] == 2641
    assert len(selected['sha256']) == 64
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['splits']['test']
    assert manifest['calibration_used'] is False
    assert manifest['config_sha256']


def test_existing_output_is_preserved(tmp_path):
    e = experiment_module()
    output = tmp_path / 'existing'
    output.mkdir()
    (output / 'important').write_text('keep')
    with pytest.raises(FileExistsError):
        e.run_cnn_experiment(tiny_config(), output)
    assert (output / 'important').read_text() == 'keep'


def test_full_tiny_run_scores_same_test_cases_for_all_methods(tmp_path):
    e = experiment_module()
    result = e.run_cnn_experiment(tiny_config(), tmp_path / 'run')
    rows = result['cases']
    assert result['manifest']['status'] == 'complete'
    assert {r['method'] for r in rows} == {'average_log', 'rician_mle', 'cnn_seed11'}
    test_ids = set(result['manifest']['splits']['test'])
    assert {r['anatomy_id'] for r in rows} == test_ids
    assert len(rows) == 12
    for key in {(r['anatomy_id'], r['delta_adc']) for r in rows}:
        subset = [r for r in rows if (r['anatomy_id'], r['delta_adc']) == key]
        assert len({r['healthy_noise_seed'] for r in subset}) == 1
        assert len({r['perturbed_noise_seed'] for r in subset}) == 1
        assert all(r['n_requested'] == r['n_valid'] + r['n_invalid'] for r in subset)
    assert result['summary']


def test_configured_change_denominator_threshold_is_used(tmp_path):
    e = experiment_module()
    c = tiny_config()
    c['training']['epochs'] = 1
    c['simulation']['recovery_epsilon'] = 1.0
    result = e.run_cnn_experiment(c, tmp_path / 'epsilon')
    assert all(r['recovery_ratio'] is None for r in result['cases'])


def test_configured_methods_and_snr_pairing_are_supported_in_reports(tmp_path):
    e = experiment_module()
    c = tiny_config()
    c['training']['epochs'] = 1
    c['simulation'].update(methods=['average_log'], snr=[5, 10, 20])
    c['evaluation']['paired_noise'] = ['independent']
    result = e.run_cnn_experiment(c, tmp_path / 'custom')
    assert result['manifest']['status'] == 'complete'
    assert {r['method'] for r in result['cases']} == {'average_log', 'cnn_seed11'}
    assert {r['snr'] for r in result['cases']} == {5, 10, 20}
    assert (tmp_path / 'custom/error_comparison.png').stat().st_size > 0
