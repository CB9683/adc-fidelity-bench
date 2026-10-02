"""Fresh grouped paired data for the spatial-context/objective comparison.

Tissue/location choices are numerical assumptions rather than biological
parameters. Generator references and masks remain separate from observations;
this module imports no neural-network runtime.
"""

import itertools
from numbers import Real

import numpy as np

from ..baselines import fit_log_linear_adc
from ..data.splits import SPLIT_NAMES
from ..pipeline import stable_seed
from ..simulation.noise import add_complex_noise
from ..simulation.partial_volume import block_average
from ..simulation.phantoms import ToyPhantom, make_toy_phantom
from ..simulation.signals import monoexponential_signal
from .data import LearningCase, _choices, _integer, learning_splits


ABLATION_KEYS = {'training_regimes', 'evaluation_regimes', 'change_loss_weight',
                 'selection_change_weight', 'null_threshold_adc', 'adc_inner_range',
                 'adc_outer_range', 's0_inner_range', 's0_outer_range'}
RANGE_KEYS = ('adc_inner_range', 'adc_outer_range', 's0_inner_range', 's0_outer_range')


def _number(value, name, *, positive=False):
    if (isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value)
            or (value <= 0 if positive else value < 0)):
        raise ValueError(f'{name} must be a finite {"positive" if positive else "nonnegative"} number')


def _geometry_position(anatomy_seed, shape):
    """Replay the existing phantom's first three geometry draws exactly."""
    rng = np.random.default_rng(anatomy_seed)
    cy = (shape[0] - 1) / 2 + rng.uniform(-0.04, 0.04) * shape[0]
    cx = (shape[1] - 1) / 2 + rng.uniform(-0.04, 0.04) * shape[1]
    return cy, cx, rng.uniform(-0.3, 0.3)


def _focal_mask(shape, cy, cx, angle, radius_pixels, focal_shape):
    y, x = np.indices(shape, dtype=float)
    xrot = (x - cx) * np.cos(angle) + (y - cy) * np.sin(angle)
    yrot = -(x - cx) * np.sin(angle) + (y - cy) * np.cos(angle)
    aspect = 1.0 if focal_shape == 'circle' else 1.5
    return ((xrot / (radius_pixels * aspect)) ** 2
            + (yrot / (radius_pixels / aspect)) ** 2 <= 1)


def _varied_location(tissue, angle, radius_pixels, focal_shape, location_seed):
    """Uniformly sample among fine-grid centers with the entire focus in tissue."""
    candidates_y, candidates_x = np.where(tissue)
    masks = _focal_mask(tissue.shape, candidates_y[:, None, None],
                        candidates_x[:, None, None], angle, radius_pixels, focal_shape)
    valid = masks.any(axis=(1, 2)) & ~np.any(masks & ~tissue[None, ...], axis=(1, 2))
    choices = np.flatnonzero(valid)
    if not len(choices):
        raise ValueError('Varied focal region has no valid fine-grid location within tissue')
    selected = choices[np.random.default_rng(location_seed).integers(len(choices))]
    return float(candidates_y[selected]), float(candidates_x[selected]), masks[selected]


def _paired_phantom(config, anatomy, regime, radius, focal_shape):
    """Construct the contrast-independent phantom and record its numerical draws."""
    sim, ablation = config['simulation'], config['ablation']
    anatomy_seed = stable_seed(sim['seed'], 'anatomy', anatomy)
    base = make_toy_phantom(anatomy_seed=anatomy_seed, shape=sim['fine_shape'],
                           voxel_spacing_mm=sim['fine_spacing_mm'], focal_radius_mm=radius,
                           delta_adc=0, focal_shape=focal_shape)
    cy, cx, angle = _geometry_position(anatomy_seed, sim['fine_shape'])
    tissue_seed = stable_seed(sim['seed'], 'ablation-tissue', anatomy, regime)
    location_seed = stable_seed(sim['seed'], 'ablation-location', anatomy, regime, radius, focal_shape)
    if regime == 'original':
        adc_inner, adc_outer, s0_inner, s0_outer = 0.0009, 0.0016, 0.8, 1.0
        baseline, s0, focal = base.adc_unperturbed, base.s0, base.focal_mask
        location_sampling = 'central_original'
    else:
        rng = np.random.default_rng(tissue_seed)
        adc_inner, adc_outer, s0_inner, s0_outer = (
            float(rng.uniform(*ablation[key])) for key in RANGE_KEYS)
        inner = base.adc_unperturbed == 0.0009
        baseline = np.where(inner, adc_inner, np.where(base.tissue_mask, adc_outer, 0.0))
        s0 = np.where(inner, s0_inner, np.where(base.tissue_mask, s0_outer, 0.0))
        cy, cx, focal = _varied_location(base.tissue_mask, angle, radius / sim['fine_spacing_mm'],
                                        focal_shape, location_seed)
        location_sampling = 'uniform_valid_fine_centers'
    phantom = ToyPhantom(baseline, baseline.copy(), s0, focal, base.tissue_mask, base.voxel_spacing_mm)
    metadata = {'regime': regime, 'anatomy_seed': anatomy_seed, 'tissue_seed': tissue_seed,
                'location_seed': location_seed, 'adc_inner': adc_inner, 'adc_outer': adc_outer,
                's0_inner': s0_inner, 's0_outer': s0_outer,
                'focal_center_y_px': float(cy), 'focal_center_x_px': float(cx),
                'focal_center_y_mm': float(cy * sim['fine_spacing_mm']),
                'focal_center_x_mm': float(cx * sim['fine_spacing_mm']),
                'focal_angle_radians': float(angle), 'location_sampling': location_sampling}
    return phantom, metadata


def ablation_splits(config):
    """Validate all settings and registered noiseless geometry before a run.

    Calibration examples remain reserved. Validation here checks geometry
    feasibility only and does not generate acquisitions or inspect model error.
    """
    if not isinstance(config, dict) or set(config) != {'schema_version', 'simulation', 'training', 'evaluation', 'ablation'}:
        raise ValueError('Ablation configuration keys must be schema_version, simulation, training, evaluation, ablation')
    ablation = config['ablation']
    if not isinstance(ablation, dict) or set(ablation) != ABLATION_KEYS:
        raise ValueError(f'Ablation keys must be exactly {sorted(ABLATION_KEYS)}')
    _choices(ablation['training_regimes'], 'training_regimes', ('varied',))
    _choices(ablation['evaluation_regimes'], 'evaluation_regimes', ('original', 'varied'))
    for key in RANGE_KEYS:
        values = ablation[key]
        if not isinstance(values, list) or len(values) != 2:
            raise ValueError(f'{key} must contain lower and upper bounds')
        for value in values:
            _number(value, key, positive=True)
        if values[0] > values[1]:
            raise ValueError(f'{key} must have ordered lower and upper bounds')
    _number(ablation['change_loss_weight'], 'change_loss_weight')
    _number(ablation['selection_change_weight'], 'selection_change_weight')
    _number(ablation['null_threshold_adc'], 'null_threshold_adc', positive=True)
    legacy = {key: value for key, value in config.items() if key != 'ablation'}
    splits = learning_splits(legacy)
    sim = config['simulation']
    if any(identifier.startswith(('toy-', 'cnn-')) for identifier in sim['anatomy_ids']):
        raise ValueError('Old toy-/cnn- anatomy IDs are excluded from the ablation cohort')
    if sim['focal_shapes'] != ['circle']:
        raise ValueError('Ablation training and validation require circle focal_shapes only')
    if 0 not in sim['delta_adc']:
        raise ValueError('Ablation contrasts must include the zero-change null control')
    if min(ablation['adc_inner_range'][0], ablation['adc_outer_range'][0]) + min(sim['delta_adc']) < 0:
        raise ValueError('Varied ADC ranges and focal contrasts can produce negative diffusivity')
    all_regimes = set(ablation['training_regimes']) | set(ablation['evaluation_regimes'])
    all_shapes = set(sim['focal_shapes']) | set(config['evaluation']['focal_shapes'])
    for anatomy, regime, radius, focal_shape in itertools.product(
            sim['anatomy_ids'], sorted(all_regimes), sim['focal_radius_mm'], sorted(all_shapes)):
        _paired_phantom(config, anatomy, regime, radius, focal_shape)
    return splits


def iter_ablation_cases(config, split, *, regimes=None, focal_shapes=None,
                        paired_noise='independent', noise_realizations=None):
    """Yield contrast-matched paired observations and separate references.

    Tissue parameters depend on anatomy/regime only; focal location additionally
    depends on size/shape. Noise draws exclude contrast from their seeds, so all
    contrasts including the null control share one draw within condition. They
    are dependent variants clustered within anatomy, not extra independent
    samples. Common pairing is an explicit variance-reducing control.
    """
    splits = ablation_splits(config)
    if split not in SPLIT_NAMES:
        raise ValueError(f'split must be drawn from {SPLIT_NAMES}')
    sim, ablation = config['simulation'], config['ablation']
    selected_regimes = (ablation['evaluation_regimes'] if split == 'test' else ablation['training_regimes']) if regimes is None else regimes
    selected_shapes = (config['evaluation']['focal_shapes'] if split == 'test' else sim['focal_shapes']) if focal_shapes is None else focal_shapes
    _choices(selected_regimes, 'regimes', ('original', 'varied'))
    _choices(selected_shapes, 'focal_shapes', ('circle', 'ellipse'))
    if split in ('train', 'validation') and (selected_regimes != ['varied'] or selected_shapes != ['circle']):
        raise ValueError('Training and validation are restricted to varied circles')
    if not set(selected_regimes) <= set(ablation['evaluation_regimes']) | set(ablation['training_regimes']):
        raise ValueError('Requested regimes must be registered in configuration')
    if not set(selected_shapes) <= set(config['evaluation']['focal_shapes']) | set(sim['focal_shapes']):
        raise ValueError('Requested focal shapes must be registered in configuration')
    if paired_noise not in ('common', 'independent'):
        raise ValueError('paired_noise must be common or independent')
    if noise_realizations is None:
        if split == 'calibration':
            raise ValueError('Reserved calibration generation requires an explicit noise_realizations count')
        noise_realizations = (config['training'][f'{split}_noise_realizations'] if split in ('train', 'validation')
                              else config['evaluation']['noise_realizations'])
    _integer(noise_realizations, 'noise_realizations')
    factor, b_values, budget = sim['partial_volume_factor'], sim['b_values'], sim['budgets'][0]
    for anatomy, regime, radius, focal_shape in itertools.product(
            splits[split], selected_regimes, sim['focal_radius_mm'], selected_shapes):
        phantom, numerical = _paired_phantom(config, anatomy, regime, radius, focal_shape)
        healthy_signal = block_average(monoexponential_signal(b_values, phantom.adc_unperturbed, phantom.s0), factor)
        healthy_reference = fit_log_linear_adc(healthy_signal, b_values).adc
        tissue_roi = block_average(phantom.tissue_mask, factor) > 0
        focal_fraction = block_average(phantom.focal_mask, factor)
        focal_roi = focal_fraction > 0
        for delta in sim['delta_adc']:
            perturbed_adc = phantom.adc_unperturbed.copy()
            perturbed_adc[phantom.focal_mask] += delta
            perturbed_signal = block_average(monoexponential_signal(b_values, perturbed_adc, phantom.s0), factor)
            reference = fit_log_linear_adc(perturbed_signal, b_values).adc
            for snr, realization in itertools.product(sim['snr'], range(noise_realizations)):
                sigma = 1.0 / snr
                parts = (anatomy, regime, radius, focal_shape, snr, realization)
                healthy_seed = stable_seed(sim['seed'], 'ablation-noise', *parts, 'healthy')
                perturbed_seed = (healthy_seed if paired_noise == 'common' else
                                  stable_seed(sim['seed'], 'ablation-noise', *parts, 'perturbed'))
                healthy_magnitude = np.abs(add_complex_noise(
                    healthy_signal, repetitions=budget, sigma=sigma, rng=np.random.default_rng(healthy_seed)))
                perturbed_magnitude = np.abs(add_complex_noise(
                    perturbed_signal, repetitions=budget, sigma=sigma, rng=np.random.default_rng(perturbed_seed)))
                metadata = {**numerical, 'anatomy_id': anatomy, 'split': split, 'snr': snr, 'sigma': sigma,
                            'budget_per_b': budget, 'total_acquisitions': len(b_values) * budget,
                            'focal_radius_mm': radius, 'focal_shape': focal_shape, 'delta_adc': delta,
                            'case_kind': 'null' if delta == 0 else 'change',
                            'noise_realization': realization, 'paired_noise': paired_noise,
                            'healthy_noise_seed': healthy_seed, 'perturbed_noise_seed': perturbed_seed,
                            'noise_seed_scope': 'shared_across_delta',
                            'focal_area_mm2': float(phantom.focal_mask.sum() * sim['fine_spacing_mm'] ** 2),
                            'focal_coarse_fraction_sum': float(focal_fraction.sum())}
                yield LearningCase(metadata, healthy_magnitude, perturbed_magnitude,
                                   healthy_reference, reference, tissue_roi, focal_roi)
