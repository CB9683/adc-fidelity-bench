"""Fixed-budget CNN training and same-acquisition quantitative comparison."""

import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..evaluation import adc_error_summary, focal_change_recovery
from ..pipeline import (_fit, _provenance, _region_error, _write_csv, _write_json,
                        aggregate_rows)
from .data import iter_learning_cases, learning_splits
from .model import SmallADCCNN, masked_scaled_mse, observed_features, predict_adc


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _examples(config, split):
    """Truth defines supervised target/loss support, never observed features."""
    features, targets, masks, registry = [], [], [], []
    for case in iter_learning_cases(config, split):
        for healthy in (True, False):
            magnitude = case.healthy_magnitude if healthy else case.perturbed_magnitude
            target = case.healthy_reference if healthy else case.reference
            features.append(observed_features(magnitude))
            targets.append(np.asarray(target, dtype=np.float32)[None])
            masks.append(case.tissue_roi[None])
            registry.append({**case.metadata, 'healthy': healthy})
    dataset = TensorDataset(torch.from_numpy(np.stack(features)),
                            torch.from_numpy(np.stack(targets)),
                            torch.from_numpy(np.stack(masks)))
    return dataset, registry


def _loss(model, loader):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for features, targets, masks in loader:
            value = masked_scaled_mse(model(features), targets, masks)
            total += float(value) * len(features)
            count += len(features)
    return total / count


def load_model_checkpoint(path, device='cpu'):
    """Load this fixed model with PyTorch's restricted tensor/state loader."""
    saved = torch.load(path, map_location=device, weights_only=True)
    if saved.get('model_spec') != 'two_channel_16_16_softplus_adc_v1':
        raise ValueError('Checkpoint model specification does not match the first CNN')
    model = SmallADCCNN().to(device)
    model.load_state_dict(saved['model_state'], strict=True)
    model.eval()
    return model


def _train_seed(config, output, train_data, validation_data, seed):
    settings = config['training']
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_data, batch_size=settings['batch_size'], shuffle=True,
                              generator=generator, num_workers=0)
    validation_loader = DataLoader(validation_data, batch_size=settings['batch_size'], shuffle=False)
    model = SmallADCCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'])
    initial_loss = _loss(model, validation_loader)
    history, best, checkpoint = [], float('inf'), output / f'checkpoint_seed{seed}.pt'
    started = time.perf_counter()
    best_epoch = None
    for epoch in range(1, settings['epochs'] + 1):
        model.train()
        total, count = 0.0, 0
        for features, targets, masks in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = masked_scaled_mse(model(features), targets, masks)
            if not torch.isfinite(loss):
                raise ValueError('Training loss became nonfinite')
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(features)
            count += len(features)
        validation_loss = _loss(model, validation_loader)
        if not np.isfinite(validation_loss):
            raise ValueError('Validation loss became nonfinite')
        improved = validation_loss < best
        if improved:
            best, best_epoch = validation_loss, epoch
            torch.save({'model_state': model.state_dict(),
                        'model_spec': 'two_channel_16_16_softplus_adc_v1',
                        'seed': seed, 'epoch': epoch, 'validation_loss': validation_loss,
                        'input_contract': 'two_magnitude_means_divided_by_observed_b0_p95',
                        'adc_units': 'mm^2/s'}, checkpoint)
        history.append({'seed': seed, 'epoch': epoch, 'train_loss': total / count,
                        'validation_loss': validation_loss, 'new_best': improved})
        if epoch == 1 or epoch % 10 == 0 or epoch == settings['epochs']:
            print(f'CNN seed {seed}: epoch {epoch}/{settings["epochs"]}; '
                  f'train MSE={total/count:.6g}; validation MSE={validation_loss:.6g}', flush=True)
    # Always evaluate the reloaded validation-selected model, not final weights.
    reloaded = load_model_checkpoint(checkpoint)
    if _loss(reloaded, validation_loader) != best:
        raise ValueError('Reloaded checkpoint does not reproduce selected validation loss')
    spec = {'seed': seed, 'filename': checkpoint.name, 'sha256': _hash(checkpoint),
            'best_epoch': best_epoch, 'validation_loss': best,
            'initial_validation_loss': initial_loss,
            'parameter_count': sum(p.numel() for p in model.parameters()),
            'training_seconds': time.perf_counter() - started}
    return spec, history


def _case_row(case, method, healthy_adc, adc, converged, healthy_converged, epsilon):
    roi, focal_roi = case.tissue_roi, case.focal_roi
    error = adc_error_summary(adc, case.reference, roi)
    healthy_error = adc_error_summary(healthy_adc, case.healthy_reference, roi)
    focal = focal_change_recovery(adc, healthy_adc, case.reference, case.healthy_reference,
                                 focal_roi, epsilon=epsilon)
    nonfocal_roi = roi & ~focal_roi
    regions = {'focal_adc': _region_error(adc, case.reference, focal_roi),
               'nonfocal_adc': _region_error(adc, case.reference, nonfocal_roi),
               'nonfocal_change': _region_error(adc - healthy_adc,
                                                case.reference - case.healthy_reference, nonfocal_roi)}
    return {**case.metadata, 'method': method, **error, 'healthy_rmse': healthy_error['rmse'],
            'healthy_n_invalid': healthy_error['n_invalid'],
            **{f'{region}_{name}': value for region, summary in regions.items()
               for name, value in summary.items()},
            **{name: focal[name] for name in ('delta_reference', 'delta_estimate', 'change_error', 'recovery_ratio')},
            **{f'focal_{name}': focal[name] for name in ('n_requested', 'n_valid', 'n_invalid')},
            'n_converged': int(np.count_nonzero(converged & roi)),
            'healthy_n_converged': int(np.count_nonzero(healthy_converged & roi)),
            'invalid_adc_fraction': error['n_invalid'] / error['n_requested'],
            'convergence_failure_fraction': float(np.mean(~converged[roi])),
            'focal_failure_fraction': float(focal['n_invalid'] > 0)}


def _evaluate(config, output, checkpoint_specs):
    simulation = config['simulation']
    models = {}
    for spec in checkpoint_specs:
        path = output / spec['filename']
        if _hash(path) != spec['sha256']:
            raise ValueError('Checkpoint changed since validation selection')
        models[f'cnn_seed{spec["seed"]}'] = load_model_checkpoint(path)
    rows, example = [], {}
    total_pairs = (len(learning_splits(config)['test']) * len(config['evaluation']['focal_shapes'])
                   * len(config['evaluation']['paired_noise']) * len(simulation['focal_radius_mm'])
                   * len(simulation['delta_adc']) * len(simulation['snr'])
                   * config['evaluation']['noise_realizations'])
    pair_index = 0
    for shape, pairing in itertools.product(config['evaluation']['focal_shapes'],
                                             config['evaluation']['paired_noise']):
        for case in iter_learning_cases(config, 'test', focal_shapes=[shape], paired_noise=pairing):
            pair_index += 1
            for method in [*simulation['methods'], *models]:
                if method in models:
                    healthy_adc = predict_adc(models[method], case.healthy_magnitude)
                    adc = predict_adc(models[method], case.perturbed_magnitude)
                    # Finite inference only: this is not a calibrated identification flag.
                    converged, healthy_converged = np.isfinite(adc), np.isfinite(healthy_adc)
                else:
                    healthy_fit = _fit(method, case.healthy_magnitude, simulation['b_values'],
                                       case.metadata['sigma'], simulation['rician_maxiter'])
                    fit = _fit(method, case.perturbed_magnitude, simulation['b_values'],
                               case.metadata['sigma'], simulation['rician_maxiter'])
                    healthy_adc = np.where(healthy_fit.valid, healthy_fit.adc, np.nan)
                    adc = np.where(fit.valid, fit.adc, np.nan)
                    converged, healthy_converged = fit.converged, healthy_fit.converged
                rows.append(_case_row(case, method, healthy_adc, adc, converged, healthy_converged,
                                      simulation['recovery_epsilon']))
                if not example:
                    example = {'reference': case.reference, 'healthy_reference': case.healthy_reference,
                               'tissue_roi': case.tissue_roi, 'focal_roi': case.focal_roi,
                               'magnitude': case.perturbed_magnitude}
                if pair_index == 1:
                    example[f'adc_{method}'] = adc
            if pair_index == 1 or pair_index % 64 == 0 or pair_index == total_pairs:
                print(f'Evaluated {pair_index}/{total_pairs} paired test acquisitions', flush=True)
    anatomy, summary = aggregate_rows(rows)
    _write_csv(output / 'cases.csv', rows)
    _write_csv(output / 'anatomy_summary.csv', anatomy)
    _write_csv(output / 'summary.csv', summary)
    np.savez_compressed(output / 'example.npz', **example)
    return rows, anatomy, summary


def _plots(output, history, anatomy, config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5), layout='constrained')
    for seed in config['training']['seeds']:
        selected = [r for r in history if r['seed'] == seed]
        ax.plot([r['epoch'] for r in selected], [r['validation_loss'] for r in selected], label=f'Seed {seed} validation')
    ax.set(xlabel='Epoch', ylabel='Per-image tissue MSE of ADC × 1000', title='Validation only — checkpoint selection')
    ax.legend(); fig.savefig(output / 'training_curve.png', dpi=150); plt.close(fig)
    if not anatomy:
        return
    methods = [*config['simulation']['methods'], *[f'cnn_seed{s}' for s in config['training']['seeds']]]
    for metric, filename, ylabel in [('rmse', 'error_comparison.png', 'ADC RMSE (10⁻³ mm²/s)'),
                                      ('recovery_ratio', 'focal_recovery.png', 'Signed focal recovery ratio'),
                                      ('invalid_adc_fraction', 'invalid_comparison.png', 'Invalid ADC fraction')]:
        snrs, pairings = config['simulation']['snr'], config['evaluation']['paired_noise']
        fig, axes = plt.subplots(len(pairings), len(snrs), squeeze=False,
                                 figsize=(6.5 * len(snrs), 4.6 * len(pairings)), layout='constrained')
        for (pairing_index, pairing), (snr_index, snr) in itertools.product(enumerate(pairings), enumerate(snrs)):
            ax = axes[pairing_index, snr_index]
            for shape in config['evaluation']['focal_shapes']:
                marker = {'circle': 'o', 'ellipse': 's'}[shape]
                values = []
                for method in methods:
                    selected = [r for r in anatomy if r['method'] == method and r['snr'] == snr
                                and r['focal_shape'] == shape and r['paired_noise'] == pairing]
                    by_id = {}
                    for r in selected:
                        if r[metric] is not None:
                            by_id.setdefault(r['anatomy_id'], []).append(r[metric])
                    means = [np.mean(v) for v in by_id.values()]
                    values.append(np.mean(means) if means else np.nan)
                if metric == 'rmse':
                    values = np.array(values) * 1000
                ax.plot(range(len(methods)), values, marker=marker, label=shape)
            if metric == 'recovery_ratio':
                ax.axhline(1, color='gray', linestyle='--', linewidth=1)
            names = {'average_log': 'Average log', 'rician_mle': 'Rician',
                     **{f'cnn_seed{s}': f'CNN {s}' for s in config['training']['seeds']}}
            ax.set_xticks(range(len(methods)), [names[method] for method in methods], rotation=30)
            ax.set(title=f'SNR {snr} — {pairing} paired noise', ylabel=ylabel)
            ax.legend()
        fig.savefig(output / filename, dpi=150); plt.close(fig)


def run_cnn_experiment(config, output_dir, *, evaluate=True):
    """Run the frozen first experiment; calibration groups are never accessed."""
    splits = learning_splits(config)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    config_path = output / 'config.json'
    _write_json(config_path, config)
    manifest = {'status': 'running', 'splits': splits, 'calibration_used': False,
                'config_sha256': _hash(config_path), 'device': 'cpu',
                'torch_version': str(torch.__version__),
                'input_contract': 'two magnitude averages / observed b0 image p95; no truth or masks',
                'target_contract': 'ADC fitted from noiseless block-averaged diffusion signals',
                'loss_contract': 'per-image tissue MSE in ADC units scaled by 1000, then equal image means',
                'checkpoint_selection': 'minimum validation loss; earliest epoch on exact tie',
                'regimes': {'training': config['simulation']['focal_shapes'],
                            'final_test': config['evaluation']['focal_shapes']},
                'limitations': ['original numerical phantoms only', 'fixed acquisition budget only',
                                'known-sigma advantage for Rician comparator only',
                                'finite CNN output is not identified or calibrated confidence',
                                'no uncertainty, stopping, clinical or scanner-time claim'],
                **_provenance()}
    _write_json(output / 'manifest.json', manifest)
    try:
        torch.set_num_threads(config['training']['num_threads'])
        torch.use_deterministic_algorithms(True)
        train, train_registry = _examples(config, 'train')
        validation, validation_registry = _examples(config, 'validation')
        _write_csv(output / 'training_examples.csv', train_registry + validation_registry)
        manifest.update(n_training_examples=len(train), n_validation_examples=len(validation),
                        checkpoints=[])
        history = []
        for seed in config['training']['seeds']:
            spec, epoch_rows = _train_seed(config, output, train, validation, seed)
            manifest['checkpoints'].append(spec)
            history.extend(epoch_rows)
            _write_csv(output / 'training_history.csv', history)
            _write_json(output / 'manifest.json', manifest)
        training_seconds = time.perf_counter() - started
        manifest.update(status='trained', training_seconds=training_seconds)
        _write_json(output / 'manifest.json', manifest)
        rows, anatomy, summary = [], [], []
        if evaluate:
            rows, anatomy, summary = _evaluate(config, output, manifest['checkpoints'])
        _plots(output, history, anatomy, config)
        manifest.update(status='complete' if evaluate else 'trained', n_cases=len(rows),
                        n_evaluated_anatomies=len({r['anatomy_id'] for r in rows}),
                        elapsed_seconds=time.perf_counter() - started)
        _write_json(output / 'manifest.json', manifest)
    except BaseException as error:
        manifest.update(status='failed', error_type=type(error).__name__,
                        elapsed_seconds=time.perf_counter() - started)
        _write_json(output / 'manifest.json', manifest)
        raise
    return {'manifest': manifest, 'cases': rows, 'anatomy_summary': anatomy, 'summary': summary}
