"""Frozen-checkpoint diagnoses use validation groups only and preserve truth units."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


def module():
    import adc_fidelity_bench.learning as learning
    assert (Path(learning.__file__).parent/'diagnostics.py').exists(), 'Development diagnostics are not implemented'
    pytest.importorskip('torch')
    from adc_fidelity_bench.learning import diagnostics
    return diagnostics


def frozen_fixture(path):
    import torch
    from adc_fidelity_bench.learning.model import SmallADCCNN
    c=json.loads((Path(__file__).parents[1]/'configs/first_cnn.json').read_text())
    c['simulation'].update(anatomy_ids=[f'cnn-diag-{i}' for i in range(4)],
        split_counts=dict(train=1,validation=1,calibration=1,test=1),fine_shape=[8,8],
        budgets=[2],snr=[20],focal_radius_mm=[0.5])
    c['training']['seeds']=[11]
    path.mkdir()
    (path/'config.json').write_text(json.dumps(c))
    torch.manual_seed(11)
    name='checkpoint_seed11.pt'
    torch.save(dict(model_state=SmallADCCNN().state_dict(),model_spec='two_channel_16_16_softplus_adc_v1'),path/name)
    m=dict(status='complete',calibration_used=False,
        config_sha256=hashlib.sha256((path/'config.json').read_bytes()).hexdigest(),
        checkpoints=[dict(seed=11,filename=name,sha256=hashlib.sha256((path/name).read_bytes()).hexdigest())])
    from adc_fidelity_bench.pipeline import _provenance
    m['source_sha256']=_provenance()['source_sha256']
    (path/'manifest.json').write_text(json.dumps(m))
    return c,m


def test_diagnosis_uses_original_validation_only_and_zero_noise_log_fit_is_exact(tmp_path):
    e=module(); c,_=frozen_fixture(tmp_path/'frozen')
    result=e.run_first_cnn_diagnostics(tmp_path/'frozen',tmp_path/'diagnosis')
    from adc_fidelity_bench.learning.data import learning_splits
    rows=result['cases']
    assert result['manifest']['status']=='complete'
    assert result['manifest']['calibration_used'] is False
    assert {r['anatomy_id'] for r in rows}==set(learning_splits(c)['validation'])
    assert {r['diagnostic_noise'] for r in rows}=={'noiseless','snr100'}
    assert len(rows)==8 and {r['method'] for r in rows}=={'average_log','cnn_seed11'}
    exact=[r for r in rows if r['diagnostic_noise']=='noiseless' and r['method']=='average_log']
    assert all(r['rmse'] < 1e-12 and abs(r['recovery_ratio']-1)<1e-10 for r in exact)
    assert all(r['perturbed_noise_seed'] != r['healthy_noise_seed'] for r in rows if r['diagnostic_noise']=='snr100')
    assert (tmp_path/'diagnosis/diagnostics.png').stat().st_size>0


def test_diagnosis_rejects_changed_checkpoint_before_creating_output(tmp_path):
    e=module(); _,m=frozen_fixture(tmp_path/'frozen')
    p=tmp_path/'frozen'/m['checkpoints'][0]['filename']; p.write_bytes(p.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='hash|changed'):
        e.run_first_cnn_diagnostics(tmp_path/'frozen',tmp_path/'diagnosis')
    assert not (tmp_path/'diagnosis').exists()


def test_diagnostic_output_is_never_overwritten(tmp_path):
    e=module(); frozen_fixture(tmp_path/'frozen')
    out=tmp_path/'diagnosis'; out.mkdir(); (out/'important').write_text('keep')
    with pytest.raises(FileExistsError): e.run_first_cnn_diagnostics(tmp_path/'frozen',out)
    assert (out/'important').read_text()=='keep'


@pytest.mark.parametrize('kind',['missing','changed'])
def test_diagnosis_requires_frozen_seed_and_split_source_hashes(tmp_path,kind):
    e=module(); _,m=frozen_fixture(tmp_path/'frozen')
    if kind=='missing': del m['source_sha256']['pipeline.py']
    else: m['source_sha256']['pipeline.py']='0'*64
    (tmp_path/'frozen/manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError,match='source|hash'):
        e.run_first_cnn_diagnostics(tmp_path/'frozen',tmp_path/'diagnosis')
    assert not (tmp_path/'diagnosis').exists()
