"""Small reproducible CPU experiments; estimator inputs contain no reference maps."""

import csv
import hashlib
import importlib.metadata
import itertools
import json
from numbers import Integral, Real
from pathlib import Path
import platform
import subprocess

import numpy as np

from .baselines import fit_log_linear_adc, fit_rician_adc
from .data.splits import SPLIT_NAMES, grouped_split
from .evaluation import adc_error_summary, focal_change_recovery
from .simulation.noise import add_complex_noise
from .simulation.partial_volume import block_average
from .simulation.phantoms import make_toy_phantom
from .simulation.signals import monoexponential_signal


CONFIG_KEYS = {
    'schema_version', 'seed', 'anatomy_ids', 'split_counts', 'evaluation_splits',
    'fine_shape', 'fine_spacing_mm', 'partial_volume_factor', 'b_values',
    'budgets', 'snr', 'focal_radius_mm', 'delta_adc', 'focal_shapes',
    'noise_realizations', 'paired_noise', 'methods', 'rician_maxiter', 'recovery_epsilon',
}
METHODS = ('average_log', 'rician_mle')
CONDITION_KEYS = ('split', 'method', 'snr', 'budget_per_b', 'total_acquisitions',
                  'focal_radius_mm', 'focal_shape', 'delta_adc', 'paired_noise')
METRIC_KEYS = ('bias', 'mae', 'rmse', 'healthy_rmse', 'delta_reference',
               'delta_estimate', 'change_error', 'recovery_ratio',
               'invalid_adc_fraction', 'convergence_failure_fraction', 'focal_failure_fraction')
METRIC_KEYS += tuple(f'{region}_{metric}' for region in ('focal_adc', 'nonfocal_adc', 'nonfocal_change')
                     for metric in ('bias', 'mae', 'rmse', 'invalid_fraction'))


def stable_seed(seed, *parts):
    """Stable 64-bit seeds independent of Python's randomized hash function."""
    payload = json.dumps([int(seed), *parts], separators=(',', ':'), ensure_ascii=True)
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], 'big')


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def _number(value, name, positive=False):
    if (isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value)
            or (positive and value <= 0)):
        raise ValueError(f'{name} must be a finite {"positive " if positive else ""}number')


def _choices(values, name, choices):
    if (not isinstance(values, list) or not values or any(not isinstance(v, str) for v in values)
            or len(set(values)) != len(values) or not set(values) <= set(choices)):
        raise ValueError(f'{name} must be a nonempty unique list drawn from {choices}')


def _numbers(values, name, *, positive=False, integer=False):
    if not isinstance(values, list) or not values:
        raise ValueError(f'{name} must be a nonempty list')
    for value in values:
        (_integer(value, name) if integer else _number(value, name, positive))
    if len(set(values)) != len(values):
        raise ValueError(f'{name} must contain unique values')


def validate_config(config):
    """Validate every generated geometry before creating result directories.

    The pilot deliberately uses two b-values, where the Rician zero-amplitude
    boundary has an explicit policy. It is not a general multi-b benchmark.
    """
    if not isinstance(config, dict) or set(config) != CONFIG_KEYS:
        raise ValueError(f'Configuration keys must be exactly {sorted(CONFIG_KEYS)}')
    _integer(config['schema_version'], 'schema_version')
    if config['schema_version'] != 1:
        raise ValueError('Only schema_version 1 is supported')
    _integer(config['seed'], 'seed', 0)
    splits = grouped_split(config['anatomy_ids'], config['split_counts'], config['seed'])
    _choices(config['evaluation_splits'], 'evaluation_splits', SPLIT_NAMES)
    if not sum(len(splits[name]) for name in config['evaluation_splits']):
        raise ValueError('Evaluation splits must contain at least one anatomy')
    shape = config['fine_shape']
    if not isinstance(shape, list) or len(shape) != 2:
        raise ValueError('fine_shape must contain two dimensions')
    for size in shape:
        _integer(size, 'fine_shape', 8)
    _integer(config['partial_volume_factor'], 'partial_volume_factor')
    if any(size % config['partial_volume_factor'] for size in shape):
        raise ValueError('fine_shape must be divisible by partial_volume_factor')
    _number(config['fine_spacing_mm'], 'fine_spacing_mm', True)
    _numbers(config['b_values'], 'b_values')
    if len(config['b_values']) != 2 or config['b_values'][0] < 0 or np.any(np.diff(config['b_values']) <= 0):
        raise ValueError('The pilot requires exactly two increasing nonnegative b-values')
    _numbers(config['budgets'], 'budgets', integer=True)
    if config['budgets'] != sorted(config['budgets']):
        raise ValueError('budgets must be increasing')
    for name in ('snr', 'focal_radius_mm'):
        _numbers(config[name], name, positive=True)
    for snr in config['snr']:
        _number(1.0 / snr, 'derived sigma', True)
    _numbers(config['delta_adc'], 'delta_adc')
    _choices(config['focal_shapes'], 'focal_shapes', ('circle', 'ellipse'))
    _choices(config['methods'], 'methods', METHODS)
    if config['paired_noise'] not in ('common', 'independent'):
        raise ValueError('paired_noise must be common or independent')
    _integer(config['noise_realizations'], 'noise_realizations')
    _integer(config['rician_maxiter'], 'rician_maxiter')
    _number(config['recovery_epsilon'], 'recovery_epsilon', True)
    for anatomy in config['anatomy_ids']:
        for radius, delta, focal_shape in itertools.product(
                config['focal_radius_mm'], config['delta_adc'], config['focal_shapes']):
            _phantom(config, anatomy, radius, delta, focal_shape)
    return splits


def _phantom(config, anatomy, radius, delta, focal_shape):
    return make_toy_phantom(
        anatomy_seed=stable_seed(config['seed'], 'anatomy', anatomy),
        shape=config['fine_shape'], voxel_spacing_mm=config['fine_spacing_mm'],
        focal_radius_mm=radius, delta_adc=delta, focal_shape=focal_shape)


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def _write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _provenance():
    source = Path(__file__).parent
    hashes = {path.relative_to(source).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(source.rglob('*.py'))}
    versions = {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'matplotlib')}
    try:
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source,
                                       stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=source,
                                            stderr=subprocess.DEVNULL, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        head, dirty = None, None
    return {'python': platform.python_version(), 'platform': platform.system(),
            'machine': platform.machine(), 'versions': versions, 'git_commit': head,
            'git_dirty': dirty, 'source_sha256': hashes}


def _fit(method, observed, b_values, sigma, maxiter):
    if method == 'average_log':
        return fit_log_linear_adc(np.mean(observed, axis=1), b_values)
    return fit_rician_adc(observed, b_values, sigma=sigma, maxiter=maxiter)


def _region_error(estimate, reference, roi):
    if roi.any():
        error = adc_error_summary(estimate, reference, roi)
        return {**error, 'invalid_fraction': error['n_invalid'] / error['n_requested']}
    return {'bias': None, 'mae': None, 'rmse': None, 'n_requested': 0,
            'n_valid': 0, 'n_invalid': 0, 'invalid_fraction': None}


def aggregate_rows(rows):
    """Average noise realizations within anatomy, then weight anatomies equally.

    Undefined metrics are excluded only with explicit numbers of contributing
    cases/anatomies. Failure fractions include every requested voxel/case.
    """
    groups = {}
    for row in rows:
        key = (row['anatomy_id'], *(row[name] for name in CONDITION_KEYS))
        groups.setdefault(key, []).append(row)
    anatomy_rows = []
    for key, cases in groups.items():
        row = {'anatomy_id': key[0], **dict(zip(CONDITION_KEYS, key[1:])), 'n_cases': len(cases)}
        for metric in METRIC_KEYS:
            values = [case[metric] for case in cases if case[metric] is not None]
            row[metric] = float(np.mean(values)) if values else None
            row[f'{metric}_n_defined_cases'] = len(values)
        anatomy_rows.append(row)
    groups = {}
    for row in anatomy_rows:
        key = tuple(row[name] for name in CONDITION_KEYS)
        groups.setdefault(key, []).append(row)
    summary = []
    for key, anatomies in groups.items():
        row = {**dict(zip(CONDITION_KEYS, key)), 'n_anatomies': len(anatomies),
               'n_cases': sum(anatomy['n_cases'] for anatomy in anatomies)}
        for metric in METRIC_KEYS:
            values = [anatomy[metric] for anatomy in anatomies if anatomy[metric] is not None]
            row[metric] = float(np.mean(values)) if values else None
            row[f'{metric}_n_defined_anatomies'] = len(values)
            row[f'{metric}_n_defined_cases'] = sum(anatomy[f'{metric}_n_defined_cases'] for anatomy in anatomies)
        summary.append(row)
    return anatomy_rows, summary


def run_benchmark(config, output_dir):
    """Run controlled paired experiments and save reproducible results.

    SNR uses reference amplitude 1, so sigma=1/SNR; local S0 varies. Complex
    noise is added after spatial signal averaging. Shared paired noise is an
    explicit variance-reducing experimental coupling, not a clinical pairing.
    """
    splits = validate_config(config)
    output = Path(output_dir)
    # Exclusive directory creation protects completed and incomplete runs alike.
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'running', 'splits': splits,
                'noise_pool_repetitions': max(config['budgets']),
                'sigma_definition': 'independent complex-component standard deviation',
                'snr_definition': 'reference amplitude 1 divided by sigma; not local tissue SNR',
                'reference_definition': 'log-linear ADC fitted to noiseless block-averaged signals',
                'mask_definition': 'coarse voxel included when fine tissue/focal fraction is greater than zero',
                'coarse_spacing_mm': config['fine_spacing_mm'] * config['partial_volume_factor'],
                'aggregation': 'noise realizations within anatomy, then equal anatomy weight; explicit undefined counts',
                'limitations': ['original numerical phantoms, not anatomical models',
                                'known-sigma Rician comparator', 'no calibrated uncertainty or stopping',
                                'no clinical or scanner-time inference'], **_provenance()}
    _write_json(output / 'config.json', config)
    _write_json(output / 'manifest.json', manifest)
    rows, example = [], {}
    factor, b_values = config['partial_volume_factor'], config['b_values']
    for split in config['evaluation_splits']:
        for anatomy in splits[split]:
            for radius, delta, focal_shape in itertools.product(
                    config['focal_radius_mm'], config['delta_adc'], config['focal_shapes']):
                phantom = _phantom(config, anatomy, radius, delta, focal_shape)
                healthy_signal = block_average(monoexponential_signal(b_values, phantom.adc_unperturbed, phantom.s0), factor)
                perturbed_signal = block_average(monoexponential_signal(b_values, phantom.adc_perturbed, phantom.s0), factor)
                healthy_reference = fit_log_linear_adc(healthy_signal, b_values).adc
                reference = fit_log_linear_adc(perturbed_signal, b_values).adc
                tissue_fraction = block_average(phantom.tissue_mask, factor)
                focal_fraction = block_average(phantom.focal_mask, factor)
                roi, focal_roi = tissue_fraction > 0, focal_fraction > 0
                for snr, realization in itertools.product(config['snr'], range(config['noise_realizations'])):
                    sigma = 1.0 / snr
                    parts = (anatomy, radius, delta, focal_shape, snr, realization)
                    healthy_seed = stable_seed(config['seed'], 'noise', *parts, 'healthy')
                    perturbed_seed = (healthy_seed if config['paired_noise'] == 'common'
                                      else stable_seed(config['seed'], 'noise', *parts, 'perturbed'))
                    healthy_pool = np.abs(add_complex_noise(healthy_signal, repetitions=max(config['budgets']), sigma=sigma,
                                                           rng=np.random.default_rng(healthy_seed)))
                    perturbed_pool = np.abs(add_complex_noise(perturbed_signal, repetitions=max(config['budgets']), sigma=sigma,
                                                             rng=np.random.default_rng(perturbed_seed)))
                    for budget in config['budgets']:
                        for method in config['methods']:
                            healthy_fit = _fit(method, healthy_pool[:, :budget], b_values, sigma, config['rician_maxiter'])
                            fit = _fit(method, perturbed_pool[:, :budget], b_values, sigma, config['rician_maxiter'])
                            healthy_adc = np.where(healthy_fit.valid, healthy_fit.adc, np.nan)
                            adc = np.where(fit.valid, fit.adc, np.nan)
                            error = adc_error_summary(adc, reference, roi)
                            healthy_error = adc_error_summary(healthy_adc, healthy_reference, roi)
                            focal = focal_change_recovery(adc, healthy_adc, reference, healthy_reference, focal_roi,
                                                         epsilon=config['recovery_epsilon'])
                            nonfocal_roi = roi & ~focal_roi
                            regional = {
                                'focal_adc': _region_error(adc, reference, focal_roi),
                                'nonfocal_adc': _region_error(adc, reference, nonfocal_roi),
                                'nonfocal_change': _region_error(adc - healthy_adc, reference - healthy_reference, nonfocal_roi),
                            }
                            row = {'anatomy_id': anatomy, 'split': split, 'method': method, 'snr': snr,
                                   'sigma': sigma, 'budget_per_b': budget, 'total_acquisitions': len(b_values) * budget,
                                   'focal_radius_mm': radius, 'focal_shape': focal_shape, 'delta_adc': delta,
                                   'noise_realization': realization, 'paired_noise': config['paired_noise'],
                                   'anatomy_seed': stable_seed(config['seed'], 'anatomy', anatomy),
                                   'healthy_noise_seed': healthy_seed, 'perturbed_noise_seed': perturbed_seed,
                                   'focal_area_mm2': float(np.sum(phantom.focal_mask) * config['fine_spacing_mm'] ** 2),
                                   'focal_coarse_fraction_sum': float(np.sum(focal_fraction)),
                                   **error, 'healthy_rmse': healthy_error['rmse'],
                                   'healthy_n_invalid': healthy_error['n_invalid'],
                                   **{f'{region}_{name}': value for region, summary in regional.items()
                                      for name, value in summary.items()},
                                   **{name: focal[name] for name in ('delta_reference', 'delta_estimate', 'change_error', 'recovery_ratio')},
                                   **{f'focal_{name}': focal[name] for name in ('n_requested', 'n_valid', 'n_invalid')},
                                   'n_converged': int(np.count_nonzero(fit.converged & roi)),
                                   'healthy_n_converged': int(np.count_nonzero(healthy_fit.converged & roi)),
                                   'invalid_adc_fraction': error['n_invalid'] / error['n_requested'],
                                   'convergence_failure_fraction': float(np.mean(~fit.converged[roi])),
                                   'focal_failure_fraction': float(focal['n_invalid'] > 0)}
                            rows.append(row)
                            if not example:
                                example = {'reference': reference, 'healthy_reference': healthy_reference,
                                           'fine_adc_unperturbed': phantom.adc_unperturbed,
                                           'fine_adc_perturbed': phantom.adc_perturbed, 'fine_s0': phantom.s0,
                                           'tissue_fraction': tissue_fraction, 'focal_fraction': focal_fraction,
                                           'noisy_magnitude': perturbed_pool[:, :budget],
                                           'adc_estimate': adc}
    anatomy_rows, summary = aggregate_rows(rows)
    _write_csv(output / 'cases.csv', rows)
    _write_csv(output / 'anatomy_summary.csv', anatomy_rows)
    _write_csv(output / 'summary.csv', summary)
    np.savez_compressed(output / 'example.npz', **example)
    from .reporting import save_plots
    save_plots(anatomy_rows, output)
    manifest.update(status='complete', n_cases=len(rows), n_evaluated_anatomies=len({r['anatomy_id'] for r in rows}),
                    example_case_index=0)
    _write_json(output / 'manifest.json', manifest)
    return {'cases': rows, 'anatomy_summary': anatomy_rows, 'summary': summary, 'manifest': manifest}
