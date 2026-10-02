"""Frozen first-CNN diagnostics on its original validation geometries only."""

from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from ..pipeline import _fit, _provenance, _write_csv, _write_json
from ..simulation.partial_volume import block_average
from ..simulation.phantoms import make_toy_phantom
from ..simulation.signals import monoexponential_signal
from .data import iter_learning_cases, learning_splits
from .experiment import _case_row, _hash, load_model_checkpoint
from .model import predict_adc


def _load_reference(reference_run):
    root=Path(reference_run)
    manifest=json.loads((root/'manifest.json').read_text())
    config=json.loads((root/'config.json').read_text())
    if manifest.get('status') != 'complete' or manifest.get('calibration_used') is not False:
        raise ValueError('Diagnostics require a completed frozen first-CNN run with calibration unused')
    if _hash(root/'config.json') != manifest.get('config_sha256'):
        raise ValueError('Reference configuration hash changed')
    splits=learning_splits(config)
    if 'splits' in manifest and manifest['splits'] != splits:
        raise ValueError('Reference split registry differs from its configuration')
    specs=manifest.get('checkpoints',[])
    if len(specs) != len(config['training']['seeds']) or {s['seed'] for s in specs} != set(config['training']['seeds']):
        raise ValueError('Reference checkpoint seeds do not match the frozen configuration')
    models={}
    for spec in specs:
        filename=spec['filename']
        if Path(filename).name != filename or not filename.endswith('.pt'):
            raise ValueError('Reference checkpoint filename must be a local .pt basename')
        path=root/filename
        if _hash(path) != spec['sha256']:
            raise ValueError('Reference checkpoint hash changed')
        models[f'cnn_seed{spec["seed"]}']=load_model_checkpoint(path)
    # CLI/orchestration additions can differ; the observed model and signal generator may not.
    current=Path(__file__).parents[1]
    essential=('__init__.py','__main__.py','baselines/__init__.py','baselines/adc.py',
        'data/__init__.py','data/splits.py','evaluation/__init__.py','evaluation/metrics.py',
        'learning/__init__.py','learning/data.py','learning/experiment.py','learning/model.py',
        'pipeline.py','reporting.py','simulation/__init__.py','simulation/phantoms.py',
        'simulation/signals.py','simulation/partial_volume.py','simulation/noise.py')
    checked=[]
    for name in essential:
        expected=manifest.get('source_sha256',{}).get(name)
        if expected is None:
            raise ValueError(f'Reference source hash is missing: {name}')
        if _hash(current/name) != expected:
            raise ValueError(f'Reference simulation/model source changed: {name}')
        checked.append(name)
    return root,config,manifest,splits,models,checked


def _noiseless_case(config,case):
    s=config['simulation']; m=case.metadata
    phantom=make_toy_phantom(anatomy_seed=m['anatomy_seed'],shape=s['fine_shape'],
        voxel_spacing_mm=s['fine_spacing_mm'],focal_radius_mm=m['focal_radius_mm'],
        delta_adc=m['delta_adc'],focal_shape=m['focal_shape'])
    magnitudes=[]
    for adc in (phantom.adc_unperturbed,phantom.adc_perturbed):
        signal=block_average(monoexponential_signal(s['b_values'],adc,phantom.s0),s['partial_volume_factor'])
        magnitudes.append(np.repeat(signal[:,None],s['budgets'][0],axis=1))
    return replace(case,healthy_magnitude=magnitudes[0],perturbed_magnitude=magnitudes[1],
        metadata={**m,'sigma':0.0,'diagnostic_noise':'noiseless','paired_noise':'noiseless'})


def _summaries(rows):
    groups={}
    for row in rows:
        key=(row['method'],row['diagnostic_noise'],row['focal_radius_mm'],row['delta_adc'])
        groups.setdefault(key,[]).append(row)
    summary=[]
    for key,cases in groups.items():
        row=dict(zip(('method','diagnostic_noise','focal_radius_mm','delta_adc'),key))
        row.update(n_anatomies=len({c['anatomy_id'] for c in cases}),n_cases=len(cases))
        for metric in ('rmse','recovery_ratio','change_error','nonfocal_change_rmse','invalid_adc_fraction'):
            defined=[r[metric] for r in cases if r[metric] is not None]
            row[metric]=float(np.mean(defined)) if defined else None
            row[f'{metric}_n_defined_cases']=len(defined)
        summary.append(row)
    return summary


def _plots(output,rows,models,config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods=['average_log',*models]
    radii=config['simulation']['focal_radius_mm']
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,noise in zip(axes,('noiseless','snr100')):
        for radius in radii:
            values=[]
            for method in methods:
                selected=[r['recovery_ratio'] for r in rows if r['method']==method
                    and r['diagnostic_noise']==noise and r['focal_radius_mm']==radius and r['recovery_ratio'] is not None]
                values.append(np.mean(selected) if selected else np.nan)
            ax.plot(range(len(methods)),values,marker='o',label=f'Radius {radius} mm')
        ax.axhline(1,color='gray',ls='--',lw=1)
        ax.set_xticks(range(len(methods)),['Average',*[m.replace('cnn_seed','CNN ') for m in models]])
        ax.set(title=noise,xlabel='Frozen method',ylabel='Signed focal recovery')
        ax.legend()
    fig.suptitle('Original validation geometries only — outside-training-noise diagnostic')
    fig.savefig(output/'diagnostics.png',dpi=150); plt.close(fig)


def run_first_cnn_diagnostics(reference_run,output_dir):
    root,config,reference,splits,models,checked=_load_reference(reference_run)
    output=Path(output_dir); output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    # A new input noise condition only; geometry/reference identity is unchanged.
    diagnostic=json.loads(json.dumps(config))
    diagnostic['simulation']['snr']=[100]
    diagnostic['training']['validation_noise_realizations']=1
    _write_json(output/'config.json',diagnostic)
    manifest={'status':'running','calibration_used':False,'role':'validation',
        'validation_ids':splits['validation'],'reference_run_name':root.name,
        'reference_manifest_sha256':_hash(root/'manifest.json'),
        'reference_config_sha256':reference['config_sha256'],
        'reference_git_commit':reference.get('git_commit'),'reference_checkpoints':reference['checkpoints'],
        'verified_essential_source_files':checked,'config_sha256':_hash(output/'config.json'),
        'device':'cpu','torch_version':str(torch.__version__),
        'conditions':['noiseless','snr100'],'noisy_pairing':'independent','noisy_draws_per_condition':1,
        'limitation':'noise conditions outside original training; development diagnostic, not cause or clinical proof',
        **_provenance()}
    _write_json(output/'manifest.json',manifest)
    try:
        torch.set_num_threads(config['training']['num_threads']); torch.use_deterministic_algorithms(True)
        rows=[]
        for noisy in iter_learning_cases(diagnostic,'validation',paired_noise='independent'):
            noisy=replace(noisy,metadata={**noisy.metadata,'diagnostic_noise':'snr100'})
            for case in (_noiseless_case(config,noisy),noisy):
                for method in ['average_log',*models]:
                    if method in models:
                        h=predict_adc(models[method],case.healthy_magnitude)
                        p=predict_adc(models[method],case.perturbed_magnitude)
                        hc,pc=np.isfinite(h),np.isfinite(p)
                    else:
                        hf=_fit(method,case.healthy_magnitude,config['simulation']['b_values'],case.metadata['sigma'],200)
                        pf=_fit(method,case.perturbed_magnitude,config['simulation']['b_values'],case.metadata['sigma'],200)
                        h=np.where(hf.valid,hf.adc,np.nan); p=np.where(pf.valid,pf.adc,np.nan)
                        hc,pc=hf.converged,pf.converged
                    row=_case_row(case,method,h,p,pc,hc,config['simulation']['recovery_epsilon'])
                    if method=='average_log' and case.metadata['diagnostic_noise']=='noiseless':
                        if row['rmse'] > 1e-12 or row['focal_n_invalid']:
                            raise ValueError('Noiseless log-fit/reference sanity check failed')
                    rows.append(row)
        summary=_summaries(rows)
        _write_csv(output/'cases.csv',rows); _write_csv(output/'summary.csv',summary)
        _plots(output,rows,models,config)
        manifest.update(status='complete',n_cases=len(rows),n_evaluated_anatomies=len(splits['validation']),
            elapsed_seconds=time.perf_counter()-started)
        _write_json(output/'manifest.json',manifest)
    except BaseException as error:
        manifest.update(status='failed',error_type=type(error).__name__,elapsed_seconds=time.perf_counter()-started)
        _write_json(output/'manifest.json',manifest)
        raise
    return {'manifest':manifest,'cases':rows,'summary':summary}
