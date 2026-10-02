"""Controlled spatial-context × objective experiment with paired null controls."""

import hashlib
import itertools
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..pipeline import CONDITION_KEYS, METRIC_KEYS, _fit, _provenance, _write_csv, _write_json
from .ablation_data import ablation_splits, iter_ablation_cases
from .ablation_model import make_ablation_model, paired_loss_components
from .experiment import _case_row, _hash
from .model import observed_features


CELLS = (
    ('spatial_mse', 'spatial', 'mse'),
    ('pointwise_mse', 'pointwise', 'mse'),
    ('spatial_change', 'spatial', 'change'),
    ('pointwise_change', 'pointwise', 'change'),
)
ABLATION_CONDITIONS = (*CONDITION_KEYS, 'regime', 'case_kind')
ABLATION_METRICS = (*METRIC_KEYS, 'focal_change_abs_error', 'opposite_sign_fraction',
    'max_tissue_abs_error', 'null_abs_change', 'null_false_change_fraction',
    'null_alarm_or_invalid_fraction')
INFERENCE_BATCH_PAIRS = 128


def paired_dataset(config, split):
    """One shared registry; supervised arrays never enter observed features."""
    arrays = [[] for _ in range(6)]
    registry = []
    for case in iter_ablation_cases(config, split):
        values = (observed_features(case.healthy_magnitude), observed_features(case.perturbed_magnitude),
                  np.asarray(case.healthy_reference, dtype=np.float32)[None],
                  np.asarray(case.reference, dtype=np.float32)[None],
                  case.tissue_roi[None], case.focal_roi[None])
        for destination, value in zip(arrays, values):
            destination.append(value)
        registry.append(dict(case.metadata))
    dataset = TensorDataset(*(torch.from_numpy(np.stack(values)) for values in arrays))
    return dataset, registry


def _components(model, batch):
    healthy, perturbed, healthy_target, target, tissue, focal = batch
    # No cross-image layer (e.g. batch normalization): each observation is inferred independently.
    outputs = model(torch.cat((healthy, perturbed), dim=0))
    hp, pp = outputs.split(len(healthy))
    return paired_loss_components(hp, pp, healthy_target, target, tissue, focal)


def validation_components(model, loader):
    model.eval()
    totals, count = {'mse': 0.0, 'change_mse': 0.0}, 0
    with torch.no_grad():
        for batch in loader:
            values = _components(model, batch)
            n_pairs = len(batch[0])
            for key in totals:
                totals[key] += float(values[key]) * n_pairs
            count += n_pairs
    return {key: value / count for key, value in totals.items()}


def load_ablation_checkpoint(path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    specifications = {'ablation_spatial_v1': 'spatial', 'ablation_pointwise_v1': 'pointwise'}
    if saved.get('model_spec') not in specifications:
        raise ValueError('Checkpoint model specification is not a registered ablation model')
    model = make_ablation_model(specifications[saved['model_spec']])
    model.load_state_dict(saved['model_state'], strict=True)
    model.eval()
    return model


def _train(config, output, train, validation, cell, architecture, objective, seed):
    settings, protocol = config['training'], config['ablation']
    torch.manual_seed(seed)
    model = make_ablation_model(architecture)
    loader = DataLoader(train, batch_size=settings['batch_size'], shuffle=True,
        generator=torch.Generator().manual_seed(seed), num_workers=0)
    val_loader = DataLoader(validation, batch_size=settings['batch_size'], shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'])
    weight = protocol['change_loss_weight'] if objective == 'change' else 0.0
    selection_weight = protocol['selection_change_weight']
    initial = validation_components(model, val_loader)
    initial_score = initial['mse'] + selection_weight * initial['change_mse']
    checkpoint = output / f'checkpoint_{cell}_seed{seed}.pt'
    best, best_epoch, best_components = float('inf'), None, None
    history = []
    started = time.perf_counter()
    for epoch in range(1, settings['epochs'] + 1):
        model.train()
        totals, count = dict(mse=0.0, change_mse=0.0, objective=0.0), 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            components = _components(model, batch)
            loss = components['mse'] + weight * components['change_mse']
            if not torch.isfinite(loss):
                raise ValueError('Training loss became nonfinite')
            loss.backward(); optimizer.step()
            n_pairs = len(batch[0])
            totals['mse'] += float(components['mse'].detach()) * n_pairs
            totals['change_mse'] += float(components['change_mse'].detach()) * n_pairs
            totals['objective'] += float(loss.detach()) * n_pairs
            count += n_pairs
        values = validation_components(model, val_loader)
        score = values['mse'] + selection_weight * values['change_mse']
        if not np.isfinite(score):
            raise ValueError('Validation score became nonfinite')
        improved = score < best
        if improved:
            best, best_epoch, best_components = score, epoch, values
            torch.save({'model_state': model.state_dict(), 'model_spec': f'ablation_{architecture}_v1',
                'seed': seed, 'cell': cell, 'objective': objective, 'epoch': epoch,
                'validation_selection_score': score, 'validation_components': values,
                'input_contract': 'two_magnitude_means_divided_by_observed_b0_p95',
                'adc_units': 'mm^2/s'}, checkpoint)
        history.append({'cell': cell, 'architecture': architecture, 'objective': objective,
            'seed': seed, 'epoch': epoch, 'train_mse': totals['mse']/count,
            'train_change_mse': totals['change_mse']/count, 'train_objective': totals['objective']/count,
            'validation_mse': values['mse'], 'validation_change_mse': values['change_mse'],
            'validation_selection_score': score, 'new_best': improved})
        if epoch == 1 or epoch % 20 == 0 or epoch == settings['epochs']:
            print(f'{cell} seed {seed}: epoch {epoch}/{settings["epochs"]}; '
                  f'validation score={score:.6g} (ADC={values["mse"]:.6g}, '
                  f'change={values["change_mse"]:.6g})', flush=True)
    reloaded = validation_components(load_ablation_checkpoint(checkpoint), val_loader)
    if reloaded != best_components:
        raise ValueError('Reloaded checkpoint does not reproduce selected validation components')
    spec = {'cell': cell, 'architecture': architecture, 'objective': objective, 'seed': seed,
        'filename': checkpoint.name, 'sha256': _hash(checkpoint), 'best_epoch': best_epoch,
        'selection_score': best, 'initial_selection_score': initial_score,
        'validation_mse': best_components['mse'], 'validation_change_mse': best_components['change_mse'],
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'training_seconds': time.perf_counter()-started}
    return spec, history


def ablation_case_row(case, method, healthy_adc, adc, converged, healthy_converged, epsilon, null_threshold):
    row = _case_row(case, method, healthy_adc, adc, converged, healthy_converged, epsilon)
    delta, reference_delta = row['delta_estimate'], row['delta_reference']
    finite = case.tissue_roi & np.isfinite(adc)
    row.update(focal_change_abs_error=None if delta is None else abs(row['change_error']),
        opposite_sign_fraction=None if delta is None or abs(reference_delta) <= epsilon
            else float(delta * reference_delta < 0),
        max_tissue_abs_error=float(np.max(np.abs(adc[finite]-case.reference[finite]))) if finite.any() else None,
        null_abs_change=None, null_false_change_fraction=None, null_alarm_or_invalid_fraction=None)
    if case.metadata['case_kind'] == 'null':
        row['null_abs_change'] = None if delta is None else abs(delta)
        row['null_false_change_fraction'] = None if delta is None else float(abs(delta) > null_threshold)
        row['null_alarm_or_invalid_fraction'] = 1.0 if delta is None else row['null_false_change_fraction']
    return row


def aggregate_ablation_rows(rows):
    """Keep regimes/null conditions distinct and retain every undefined count."""
    groups = {}
    for row in rows:
        key = (row['anatomy_id'], *(row[name] for name in ABLATION_CONDITIONS))
        groups.setdefault(key, []).append(row)
    anatomy = []
    for key, cases in groups.items():
        row = {'anatomy_id': key[0], **dict(zip(ABLATION_CONDITIONS,key[1:])), 'n_cases': len(cases)}
        for metric in ABLATION_METRICS:
            values = [r[metric] for r in cases if r[metric] is not None]
            row[metric] = float(np.mean(values)) if values else None
            row[f'{metric}_n_defined_cases'] = len(values)
        anatomy.append(row)
    groups = {}
    for row in anatomy:
        groups.setdefault(tuple(row[name] for name in ABLATION_CONDITIONS), []).append(row)
    summary = []
    for key, anatomies in groups.items():
        row = {**dict(zip(ABLATION_CONDITIONS,key)), 'n_anatomies': len(anatomies),
            'n_cases': sum(r['n_cases'] for r in anatomies)}
        for metric in ABLATION_METRICS:
            values = [r[metric] for r in anatomies if r[metric] is not None]
            row[metric] = float(np.mean(values)) if values else None
            row[f'{metric}_n_defined_anatomies'] = len(values)
            row[f'{metric}_n_defined_cases'] = sum(r[f'{metric}_n_defined_cases'] for r in anatomies)
        summary.append(row)
    return anatomy, summary


def _evaluate(config, output, specs):
    models = {}
    for spec in specs:
        path = output / spec['filename']
        if _hash(path) != spec['sha256']:
            raise ValueError('Checkpoint changed after validation selection')
        models[f'{spec["cell"]}_seed{spec["seed"]}'] = load_ablation_checkpoint(path)
    rows, example, fit_cache, pair_count = [], {}, {}, 0
    settings = config['simulation']
    total = (len(ablation_splits(config)['test']) * len(config['ablation']['evaluation_regimes'])
        * len(config['evaluation']['focal_shapes']) * len(config['evaluation']['paired_noise'])
        * len(settings['snr']) * len(settings['focal_radius_mm']) * len(settings['delta_adc'])
        * config['evaluation']['noise_realizations'])
    for regime, shape, pairing in itertools.product(config['ablation']['evaluation_regimes'],
            config['evaluation']['focal_shapes'], config['evaluation']['paired_noise']):
        iterator = iter_ablation_cases(config, 'test', regimes=[regime], focal_shapes=[shape], paired_noise=pairing)
        while True:
            cases = list(itertools.islice(iterator, INFERENCE_BATCH_PAIRS))
            if not cases:
                break
            healthy_features = [observed_features(c.healthy_magnitude) for c in cases]
            perturbed_features = [observed_features(c.perturbed_magnitude) for c in cases]
            inputs = torch.from_numpy(np.stack(healthy_features + perturbed_features))
            predictions = {}
            with torch.no_grad():
                for method, model in models.items():
                    predictions[method] = model(inputs)[:,0].cpu().numpy()
            for i, case in enumerate(cases):
                pair_count += 1
                healthy_digest = hashlib.sha256(case.healthy_magnitude.tobytes()).hexdigest()
                perturbed_digest = hashlib.sha256(case.perturbed_magnitude.tobytes()).hexdigest()
                for method in [*settings['methods'], *models]:
                    if method in models:
                        healthy_adc, adc = predictions[method][i], predictions[method][len(cases)+i]
                        converged, healthy_converged = np.isfinite(adc), np.isfinite(healthy_adc)
                    else:
                        fitted = []
                        for magnitude, digest in [(case.healthy_magnitude,healthy_digest),
                                                   (case.perturbed_magnitude,perturbed_digest)]:
                            key = (method,case.metadata['sigma'],magnitude.shape,digest)
                            if key not in fit_cache:
                                fit_cache[key] = _fit(method,magnitude,settings['b_values'],
                                    case.metadata['sigma'],settings['rician_maxiter'])
                            fitted.append(fit_cache[key])
                        healthy_fit, fit = fitted
                        healthy_adc = np.where(healthy_fit.valid, healthy_fit.adc, np.nan)
                        adc = np.where(fit.valid, fit.adc, np.nan)
                        converged, healthy_converged = fit.converged, healthy_fit.converged
                    rows.append(ablation_case_row(case,method,healthy_adc,adc,converged,healthy_converged,
                        settings['recovery_epsilon'],config['ablation']['null_threshold_adc']))
                    if pair_count == 1:
                        example.update(reference=case.reference, healthy_reference=case.healthy_reference,
                            tissue_roi=case.tissue_roi, focal_roi=case.focal_roi,
                            magnitude=case.perturbed_magnitude, healthy_magnitude=case.healthy_magnitude)
                        example[f'adc_{method}'] = adc
                        example[f'healthy_adc_{method}'] = healthy_adc
            print(f'Evaluated {pair_count}/{total} paired acquisitions',flush=True)
    anatomy, summary = aggregate_ablation_rows(rows)
    _write_csv(output/'cases.csv',rows); _write_csv(output/'anatomy_summary.csv',anatomy)
    _write_csv(output/'summary.csv',summary)
    np.savez_compressed(output/'example.npz',**example)
    return rows, anatomy, summary


def _compact(anatomy, metric, **filters):
    by_id = {}
    for row in anatomy:
        if all(row[key] == value for key,value in filters.items()) and row[metric] is not None:
            by_id.setdefault(row['anatomy_id'],[]).append(row[metric])
    means = [np.mean(values) for values in by_id.values()]
    return float(np.mean(means)) if means else np.nan


def _plots(output, history, anatomy, config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for ax,(cell,_,_) in zip(axes.flat,CELLS):
        for seed in config['training']['seeds']:
            selected = [r for r in history if r['cell']==cell and r['seed']==seed]
            ax.plot([r['epoch'] for r in selected],[r['validation_selection_score'] for r in selected],label=f'Seed {seed}')
        ax.set(title=cell.replace('_',' '),xlabel='Epoch',
            ylabel=f'Validation ADC MSE + {config["ablation"]["selection_change_weight"]:g} × change MSE')
        ax.legend()
    fig.savefig(output/'training_curve.png',dpi=150); plt.close(fig)
    if not anatomy:
        return
    methods = [*config['simulation']['methods'],
        *[f'{cell}_seed{seed}' for cell,_,_ in CELLS for seed in config['training']['seeds']]]
    short = {'average_log':'Average','rician_mle':'Rice'}
    for cell,architecture,objective in CELLS:
        for seed in config['training']['seeds']:
            short[f'{cell}_seed{seed}'] = f'{"S" if architecture=="spatial" else "P"}-{"MSE" if objective=="mse" else "Δ"} {seed}'
    regimes, snrs = config['ablation']['evaluation_regimes'], config['simulation']['snr']
    pairings = config['evaluation']['paired_noise']
    primary = 'independent' if 'independent' in pairings else pairings[0]
    colors = {'average_log':'C4','rician_mle':'C5', **{cell:f'C{i}' for i,(cell,_,_) in enumerate(CELLS)}}
    fig, axes = plt.subplots(len(regimes),len(snrs),squeeze=False,
        figsize=(6*len(snrs),4.5*len(regimes)),layout='constrained')
    for (ri,regime),(si,snr) in itertools.product(enumerate(regimes),enumerate(snrs)):
        ax=axes[ri,si]
        labels=set()
        for method in methods:
            label=method.split('_seed')[0]
            for shape in config['evaluation']['focal_shapes']:
                filters=dict(method=method,regime=regime,snr=snr,paired_noise=primary,focal_shape=shape,case_kind='change')
                x=_compact(anatomy,'rmse',**filters)*1000
                y=_compact(anatomy,'recovery_ratio',**filters)
                ax.scatter(x,y,color=colors[label],marker='o' if shape=='circle' else 's',
                    label=label.replace('_',' ') if label not in labels else None,alpha=.8)
                labels.add(label)
        ax.axhline(1,color='gray',ls='--',lw=1)
        ax.set(title=f'{regime} — SNR {snr}, {primary} noise',xlabel='Conditional tissue RMSE (10⁻³ mm²/s)',
            ylabel='Signed focal recovery (change cases)')
        ax.legend(fontsize=8)
    fig.suptitle('Noise reduction versus focal preservation; circles ○, ellipses □; all seeds shown')
    fig.savefig(output/'tradeoff.png',dpi=150); plt.close(fig)
    for metric,filename,ylabel in [('recovery_ratio','focal_recovery.png','Signed focal recovery'),
                                  ('invalid_adc_fraction','invalid_comparison.png','Invalid tissue ADC fraction')]:
        fig,axes=plt.subplots(len(regimes)*len(pairings),len(snrs),squeeze=False,
            figsize=(7*len(snrs),4*len(regimes)*len(pairings)),layout='constrained')
        for row_index,(regime,pairing) in enumerate(itertools.product(regimes,pairings)):
            for si,snr in enumerate(snrs):
                ax=axes[row_index,si]
                for shape in config['evaluation']['focal_shapes']:
                    extra={'case_kind':'change'} if metric=='recovery_ratio' else {}
                    values=[_compact(anatomy,metric,method=m,regime=regime,snr=snr,
                        paired_noise=pairing,focal_shape=shape,**extra) for m in methods]
                    ax.plot(range(len(methods)),values,marker='o' if shape=='circle' else 's',label=shape)
                if metric=='recovery_ratio': ax.axhline(1,color='gray',ls='--',lw=1)
                ax.set_xticks(range(len(methods)),[short[m] for m in methods],rotation=50,ha='right')
                ax.set(title=f'{regime} — SNR {snr}, {pairing}',ylabel=ylabel)
                ax.legend()
        fig.suptitle('S: spatial; P: pointwise; MSE: tissue loss; Δ: tissue + paired-change loss')
        fig.savefig(output/filename,dpi=150); plt.close(fig)
    fig,axes=plt.subplots(len(regimes),len(snrs),squeeze=False,
        figsize=(7*len(snrs),4.5*len(regimes)),layout='constrained')
    for (ri,regime),(si,snr) in itertools.product(enumerate(regimes),enumerate(snrs)):
        ax=axes[ri,si]
        for metric,label,linestyle in [('null_false_change_fraction','False change among defined pairs','--'),
            ('null_alarm_or_invalid_fraction','False change OR invalid; all pairs','-')]:
            values=[_compact(anatomy,metric,method=m,regime=regime,snr=snr,
                paired_noise=primary,case_kind='null') for m in methods]
            ax.plot(range(len(methods)),values,marker='o',ls=linestyle,label=label)
        ax.set_xticks(range(len(methods)),[short[m] for m in methods],rotation=50,ha='right')
        ax.set(title=f'{regime} — SNR {snr}, {primary}',ylabel='Null-case fraction')
        ax.legend(fontsize=8)
    fig.suptitle(f'Healthy controls: |estimated focal change| > {config["ablation"]["null_threshold_adc"]:.3g} mm²/s; no detection/clinical claim')
    fig.savefig(output/'null_controls.png',dpi=150); plt.close(fig)


def run_ablation(config, output_dir, *, evaluate=True):
    splits=ablation_splits(config)
    output=Path(output_dir); output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    _write_json(output/'config.json',config)
    manifest={'status':'running','splits':splits,'calibration_used':False,
        'config_sha256':_hash(output/'config.json'),'device':'cpu','torch_version':str(torch.__version__),
        'input_contract':'two observed magnitude means / observed b0 p95; no truth or masks',
        'target_contract':'ADC from noiseless spatially averaged signals',
        'selection_contract':'same minimum validation ADC MSE + selection_weight*paired change MSE; earliest exact tie',
        'training_contract':'same paired data/shuffle per seed; tissue MSE with change weight 0 or frozen weight',
        'batch_unit':'healthy/perturbed pairs','inference_batch_pairs':INFERENCE_BATCH_PAIRS,
        'null_threshold_adc':config['ablation']['null_threshold_adc'],
        'primary_pairing':'independent' if 'independent' in config['evaluation']['paired_noise'] else config['evaluation']['paired_noise'][0],
        'selection_change_weight':config['ablation']['selection_change_weight'],
        'change_loss_weight':config['ablation']['change_loss_weight'],'checkpoints':[],
        'limitations':['restricted original numerical geometries','same observed global normalization for all cells',
            'known-sigma advantage for Rician only','finite outputs are not identified or calibrated confidence',
            'no stopping, acquisition reduction, clinical or causal-mechanism guarantee'],**_provenance()}
    _write_json(output/'manifest.json',manifest)
    try:
        torch.set_num_threads(config['training']['num_threads']); torch.use_deterministic_algorithms(True)
        train,train_registry=paired_dataset(config,'train')
        validation,val_registry=paired_dataset(config,'validation')
        _write_csv(output/'training_pairs.csv',train_registry+val_registry)
        manifest.update(n_training_pairs=len(train),n_validation_pairs=len(validation),
            n_training_images=2*len(train),n_validation_images=2*len(validation))
        history=[]
        for cell,architecture,objective in CELLS:
            for seed in config['training']['seeds']:
                spec,records=_train(config,output,train,validation,cell,architecture,objective,seed)
                manifest['checkpoints'].append(spec); history.extend(records)
                _write_csv(output/'training_history.csv',history); _write_json(output/'manifest.json',manifest)
        manifest.update(status='trained',training_seconds=time.perf_counter()-started)
        _write_json(output/'manifest.json',manifest)
        rows,anatomy,summary=[],[],[]
        if evaluate: rows,anatomy,summary=_evaluate(config,output,manifest['checkpoints'])
        _plots(output,history,anatomy,config)
        manifest.update(status='complete' if evaluate else 'trained',n_cases=len(rows),
            n_evaluated_anatomies=len({r['anatomy_id'] for r in rows}),elapsed_seconds=time.perf_counter()-started)
        _write_json(output/'manifest.json',manifest)
    except BaseException as error:
        manifest.update(status='failed',error_type=type(error).__name__,elapsed_seconds=time.perf_counter()-started)
        _write_json(output/'manifest.json',manifest)
        raise
    return {'manifest':manifest,'cases':rows,'anatomy_summary':anatomy,'summary':summary}
