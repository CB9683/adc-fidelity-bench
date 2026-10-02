"""Actual paired learning, selection and fresh matched evaluation."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest


def tiny_config():
    c = json.loads((Path(__file__).parents[1] / 'configs/first_cnn.json').read_text())
    c['simulation'].update(anatomy_ids=[f'abl-smoke-{i}' for i in range(4)],
        split_counts=dict(train=1, validation=1, calibration=1, test=1),
        fine_shape=[8,8], budgets=[2], snr=[20], focal_radius_mm=[0.5],
        delta_adc=[-0.0003,0,0.0003])
    c['training'].update(seeds=[11], epochs=3, batch_size=8, learning_rate=0.003,
        train_noise_realizations=1, validation_noise_realizations=1)
    c['evaluation'].update(focal_shapes=['circle'], paired_noise=['independent','common'], noise_realizations=1)
    c['ablation'] = dict(training_regimes=['varied'], evaluation_regimes=['original','varied'],
        change_loss_weight=1.0, selection_change_weight=1.0, null_threshold_adc=1e-4,
        adc_inner_range=[0.0007,0.0012], adc_outer_range=[0.0013,0.0019],
        s0_inner_range=[0.65,0.95], s0_outer_range=[0.85,1.15])
    return c


def module():
    import adc_fidelity_bench.learning as learning
    assert (Path(learning.__file__).parent / 'ablation.py').exists(), 'Ablation runner is not implemented'
    pytest.importorskip('torch')
    from adc_fidelity_bench.learning import ablation
    return ablation


def test_all_four_cells_genuinely_train_and_use_common_validation_selection(tmp_path):
    e = module()
    result = e.run_ablation(tiny_config(), tmp_path/'train', evaluate=False)
    m = result['manifest']
    assert m['status']=='trained' and m['calibration_used'] is False
    assert len(m['checkpoints'])==4
    history = list(csv.DictReader((tmp_path/'train/training_history.csv').open()))
    assert len(history)==12
    scores=[]
    for spec in m['checkpoints']:
        rows=[r for r in history if r['cell']==spec['cell'] and int(r['seed'])==spec['seed']]
        assert spec['selection_score']==min(float(r['validation_selection_score']) for r in rows)
        assert spec['best_epoch']==min(int(r['epoch']) for r in rows if float(r['validation_selection_score'])==spec['selection_score'])
        assert all(float(r['validation_selection_score'])==pytest.approx(float(r['validation_mse'])+float(r['validation_change_mse'])) for r in rows)
        scores.append(spec['selection_score'] < spec['initial_selection_score'])
        assert spec['parameter_count']==(2641 if spec['architecture']=='spatial' else 2647)
        a=e.load_ablation_checkpoint(tmp_path/'train'/spec['filename'])
        b=e.load_ablation_checkpoint(tmp_path/'train'/spec['filename'])
        from adc_fidelity_bench.learning.model import predict_adc
        np.testing.assert_array_equal(predict_adc(a,np.ones((2,2,4,4))),predict_adc(b,np.ones((2,2,4,4))))
    assert any(scores), 'Optimization must improve at least one real validation score'
    assert m['n_training_pairs']==3 and m['n_validation_pairs']==3
    assert len(m['config_sha256'])==64 and m['source_sha256']


def test_existing_output_is_preserved(tmp_path):
    e=module()
    out=tmp_path/'existing'; out.mkdir(); (out/'important').write_text('keep')
    with pytest.raises(FileExistsError): e.run_ablation(tiny_config(),out)
    assert (out/'important').read_text()=='keep'


def test_configured_pairing_and_selection_weight_are_recorded_truthfully(tmp_path):
    e=module(); c=tiny_config(); c['training']['epochs']=1
    c['evaluation']['paired_noise']=['common']
    c['ablation']['selection_change_weight']=2.0
    result=e.run_ablation(c,tmp_path/'custom',evaluate=False)
    assert result['manifest']['primary_pairing']=='common'
    assert result['manifest']['selection_change_weight']==2.0
    for spec in result['manifest']['checkpoints']:
        assert spec['selection_score']==pytest.approx(spec['validation_mse']+2*spec['validation_change_mse'])


def test_every_cell_and_baseline_uses_identical_test_pairs_and_null_denominators(tmp_path):
    e=module(); c=tiny_config(); c['training']['epochs']=1
    result=e.run_ablation(c,tmp_path/'run')
    m=result['manifest']; rows=result['cases']
    assert m['status']=='complete' and m['calibration_used'] is False
    assert len(rows)==72 and m['n_evaluated_anatomies']==1
    assert len({r['method'] for r in rows})==6
    assert {r['anatomy_id'] for r in rows}==set(m['splits']['test'])
    keys={(r['anatomy_id'],r['regime'],r['paired_noise'],r['delta_adc']) for r in rows}
    for key in keys:
        cases=[r for r in rows if (r['anatomy_id'],r['regime'],r['paired_noise'],r['delta_adc'])==key]
        assert len({r['healthy_noise_seed'] for r in cases})==1
        assert len({r['perturbed_noise_seed'] for r in cases})==1
        assert len(cases)==6
    null=[r for r in rows if r['case_kind']=='null']
    assert len(null)==24 and all(r['recovery_ratio'] is None for r in null)
    assert all(r['null_alarm_or_invalid_fraction'] in (0.0,1.0) for r in null)
    assert all(r['null_alarm_or_invalid_fraction'] is None for r in rows if r['case_kind']=='change')
    for r in null:
        if r['focal_n_invalid']: assert r['null_alarm_or_invalid_fraction']==1
        else: assert r['null_alarm_or_invalid_fraction']==float(abs(r['delta_estimate']) > c['ablation']['null_threshold_adc'])
    assert {r['regime'] for r in result['anatomy_summary']}=={'original','varied'}
    assert {r['case_kind'] for r in result['summary']}=={'null','change'}
    assert all(r['n_requested']==r['n_valid']+r['n_invalid'] for r in rows)
    for name in ['tradeoff.png','focal_recovery.png','null_controls.png','invalid_comparison.png']:
        assert (tmp_path/'run'/name).stat().st_size>0


def test_null_metrics_charge_invalid_focal_pair_without_inventing_recovery():
    e=module()
    from adc_fidelity_bench.learning.ablation_data import iter_ablation_cases
    case=next(c for c in iter_ablation_cases(tiny_config(),'test') if c.metadata['case_kind']=='null')
    a=case.reference.copy(); h=case.healthy_reference.copy(); a[case.focal_roi]=np.nan
    row=e.ablation_case_row(case,'failed',h,a,np.isfinite(a),np.isfinite(h),1e-8,1e-4)
    assert row['null_alarm_or_invalid_fraction']==1.0
    assert row['null_false_change_fraction'] is None
    assert row['null_abs_change'] is None and row['recovery_ratio'] is None


def test_aggregation_does_not_mix_regimes_or_noise_draws_as_independent_anatomies():
    e=module()
    from adc_fidelity_bench.learning.ablation_data import iter_ablation_cases
    case=next(iter_ablation_cases(tiny_config(),'test'))
    a=case.reference.copy(); h=case.healthy_reference.copy()
    base=e.ablation_case_row(case,'oracle',h,a,np.isfinite(a),np.isfinite(h),1e-8,1e-4)
    rows=[{**base,'regime':'original','noise_realization':0,'rmse':1.0},
          {**base,'regime':'original','noise_realization':1,'rmse':3.0},
          {**base,'regime':'varied','noise_realization':0,'rmse':10.0}]
    anatomy,summary=e.aggregate_ablation_rows(rows)
    assert len(anatomy)==2 and len(summary)==2
    original=next(r for r in summary if r['regime']=='original')
    assert original['rmse']==2.0 and original['n_anatomies']==1 and original['n_cases']==2
