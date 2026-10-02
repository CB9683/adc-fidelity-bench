"""Grouped numerical learning examples; truth is for loss and evaluation only.

This module does not import the optional neural-network runtime. Magnitudes
follow the same acquisition simulation as the conventional benchmark, and
all derived variants inherit their anatomy's preassigned role.
"""

from dataclasses import dataclass
import itertools
from numbers import Integral, Real

import numpy as np

from ..baselines import fit_log_linear_adc
from ..data.splits import SPLIT_NAMES
from ..pipeline import stable_seed, validate_config
from ..simulation.noise import add_complex_noise
from ..simulation.partial_volume import block_average
from ..simulation.phantoms import make_toy_phantom
from ..simulation.signals import monoexponential_signal


WRAPPER_KEYS = {'schema_version', 'simulation', 'training', 'evaluation'}
TRAINING_KEYS = {'seeds', 'epochs', 'batch_size', 'learning_rate', 'device',
                 'num_threads', 'train_noise_realizations', 'validation_noise_realizations'}
EVALUATION_KEYS = {'focal_shapes', 'paired_noise', 'noise_realizations'}


@dataclass(frozen=True)
class LearningCase:
    """One paired acquisition with separate training/evaluation references.

    Measurements have shape ``(b_value, repetition, height, width)``. The
    references are ADC fitted to noiseless coarse signals, with NaN outside
    signal support. ROIs are generator truth and must never enter model inputs
    or prediction-time preprocessing.
    """

    metadata: dict
    healthy_magnitude: np.ndarray
    perturbed_magnitude: np.ndarray
    healthy_reference: np.ndarray
    reference: np.ndarray
    tissue_roi: np.ndarray
    focal_roi: np.ndarray


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def _choices(values, name, allowed):
    if (not isinstance(values, list) or not values
            or any(not isinstance(value, str) for value in values)
            or len(set(values)) != len(values) or not set(values) <= set(allowed)):
        raise ValueError(f'{name} must be a nonempty unique list drawn from {allowed}')


def learning_splits(config):
    """Validate the fixed two-b learning task and allocate independent roles.

    A single acquisition budget is required because this first small CNN is
    trained for a fixed repeat count. Small smoke experiments may use another
    positive budget, but retain b=[0, 1000] and all four nonempty anatomy roles.
    The already-inspected toy pilot geometries cannot enter this new cohort.
    """
    if not isinstance(config, dict) or set(config) != WRAPPER_KEYS:
        raise ValueError(f'Learning configuration keys must be exactly {sorted(WRAPPER_KEYS)}')
    _integer(config['schema_version'], 'schema_version')
    if config['schema_version'] != 1:
        raise ValueError('Only learning schema_version 1 is supported')
    simulation = config['simulation']
    splits = validate_config(simulation)
    if simulation['b_values'] != [0, 1000]:
        raise ValueError('The first CNN requires fixed b_values [0, 1000]')
    if len(simulation['budgets']) != 1:
        raise ValueError('The first CNN requires exactly one fixed acquisition budget')
    if any(not splits[name] for name in SPLIT_NAMES):
        raise ValueError('Every learning anatomy role must be nonempty')
    if any(identifier.startswith('toy-') for identifier in simulation['anatomy_ids']):
        raise ValueError('Old toy- pilot anatomy IDs must be excluded from the learning cohort')
    training = config['training']
    if not isinstance(training, dict) or set(training) != TRAINING_KEYS:
        raise ValueError(f'Training keys must be exactly {sorted(TRAINING_KEYS)}')
    seeds = training['seeds']
    if not isinstance(seeds, list) or not seeds:
        raise ValueError('Training seeds must be a nonempty unique list')
    for seed in seeds:
        _integer(seed, 'training seed', minimum=0)
    if len(set(seeds)) != len(seeds):
        raise ValueError('Training seeds must be unique')
    for name in ('epochs', 'batch_size', 'num_threads', 'train_noise_realizations',
                 'validation_noise_realizations'):
        _integer(training[name], name)
    learning_rate = training['learning_rate']
    if (isinstance(learning_rate, bool) or not isinstance(learning_rate, Real)
            or not np.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError('learning_rate must be finite and positive')
    if training['device'] != 'cpu':
        raise ValueError('The reproducible first CNN experiment uses device cpu')
    evaluation = config['evaluation']
    if not isinstance(evaluation, dict) or set(evaluation) != EVALUATION_KEYS:
        raise ValueError(f'Evaluation keys must be exactly {sorted(EVALUATION_KEYS)}')
    _choices(evaluation['focal_shapes'], 'evaluation focal_shapes', ('circle', 'ellipse'))
    _choices(evaluation['paired_noise'], 'evaluation paired_noise', ('common', 'independent'))
    _integer(evaluation['noise_realizations'], 'evaluation noise_realizations')
    # Validate held-out shape geometry before allocating any training output.
    # The shallow copy is sufficient because validate_config only reads it;
    # replacing the shape list must not change the training shape family.
    evaluation_simulation = {**simulation, 'focal_shapes': list(evaluation['focal_shapes'])}
    validate_config(evaluation_simulation)
    return splits


def iter_learning_cases(config, split, focal_shapes=None, paired_noise='independent',
                        noise_realizations=None):
    """Yield deterministic examples without coupling truth to model code.

    Training defaults to its configured four noise draws and validation to
    two, while test defaults to the evaluation draw count. Calibration is
    reserved: generating its cases requires an explicit draw count. Override
    focal shapes explicitly to evaluate the held-out elliptical shape family.
    Paired noise is independent by default; common noise is only a requested
    control and preserves identical noise outside the focal signal footprint.
    """
    splits = learning_splits(config)
    if split not in SPLIT_NAMES:
        raise ValueError(f'split must be drawn from {SPLIT_NAMES}')
    simulation = config['simulation']
    shapes = simulation['focal_shapes'] if focal_shapes is None else focal_shapes
    _choices(shapes, 'focal_shapes', ('circle', 'ellipse'))
    if paired_noise not in ('common', 'independent'):
        raise ValueError('paired_noise must be common or independent')
    if noise_realizations is None:
        if split == 'calibration':
            raise ValueError('Reserved calibration generation requires an explicit noise_realizations count')
        if split == 'train':
            noise_realizations = config['training']['train_noise_realizations']
        elif split == 'validation':
            noise_realizations = config['training']['validation_noise_realizations']
        else:
            noise_realizations = config['evaluation']['noise_realizations']
    _integer(noise_realizations, 'noise_realizations')
    factor, b_values = simulation['partial_volume_factor'], simulation['b_values']
    budget = simulation['budgets'][0]
    for anatomy in splits[split]:
        anatomy_seed = stable_seed(simulation['seed'], 'anatomy', anatomy)
        for radius, delta, shape in itertools.product(
                simulation['focal_radius_mm'], simulation['delta_adc'], shapes):
            phantom = make_toy_phantom(
                anatomy_seed=anatomy_seed, shape=simulation['fine_shape'],
                voxel_spacing_mm=simulation['fine_spacing_mm'],
                focal_radius_mm=radius, delta_adc=delta, focal_shape=shape)
            healthy_signal = block_average(monoexponential_signal(
                b_values, phantom.adc_unperturbed, phantom.s0), factor)
            perturbed_signal = block_average(monoexponential_signal(
                b_values, phantom.adc_perturbed, phantom.s0), factor)
            healthy_reference = fit_log_linear_adc(healthy_signal, b_values).adc
            reference = fit_log_linear_adc(perturbed_signal, b_values).adc
            tissue_roi = block_average(phantom.tissue_mask, factor) > 0
            focal_fraction = block_average(phantom.focal_mask, factor)
            focal_roi = focal_fraction > 0
            for snr, realization in itertools.product(simulation['snr'], range(noise_realizations)):
                sigma = 1.0 / snr
                parts = (anatomy, radius, delta, shape, snr, realization)
                healthy_seed = stable_seed(simulation['seed'], 'noise', *parts, 'healthy')
                perturbed_seed = (healthy_seed if paired_noise == 'common' else
                                  stable_seed(simulation['seed'], 'noise', *parts, 'perturbed'))
                healthy_magnitude = np.abs(add_complex_noise(
                    healthy_signal, repetitions=budget, sigma=sigma,
                    rng=np.random.default_rng(healthy_seed)))
                perturbed_magnitude = np.abs(add_complex_noise(
                    perturbed_signal, repetitions=budget, sigma=sigma,
                    rng=np.random.default_rng(perturbed_seed)))
                metadata = {
                    'anatomy_id': anatomy, 'split': split, 'snr': snr, 'sigma': sigma,
                    'budget_per_b': budget, 'total_acquisitions': len(b_values) * budget,
                    'focal_radius_mm': radius, 'focal_shape': shape, 'delta_adc': delta,
                    'noise_realization': realization, 'paired_noise': paired_noise,
                    'anatomy_seed': anatomy_seed, 'healthy_noise_seed': healthy_seed,
                    'perturbed_noise_seed': perturbed_seed,
                    'focal_area_mm2': float(phantom.focal_mask.sum() * simulation['fine_spacing_mm'] ** 2),
                    'focal_coarse_fraction_sum': float(focal_fraction.sum()),
                }
                yield LearningCase(metadata, healthy_magnitude, perturbed_magnitude,
                                   healthy_reference, reference, tissue_roi, focal_roi)
