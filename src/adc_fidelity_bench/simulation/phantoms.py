"""Original geometric phantoms with synthetic, non-biological parameters."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class ToyPhantom:
    adc_unperturbed: np.ndarray
    adc_perturbed: np.ndarray
    s0: np.ndarray
    focal_mask: np.ndarray
    tissue_mask: np.ndarray
    voxel_spacing_mm: float


def make_toy_phantom(*, anatomy_seed, shape, voxel_spacing_mm,
                     focal_radius_mm, delta_adc, focal_shape='circle'):
    """Generate a varied two-region ellipse and a known signed focal change.

    Synthetic tissue ADC is 0.0009/0.0016 mm^2/s; S0 is 0.8/1.0.
    These values define a numerical toy experiment, not normal tissue ranges.
    Geometry is determined only by ``anatomy_seed``. Elliptical focal lesions
    use major/minor axis ratio 2.25 with the same nominal area as a radius-matched circle.
    """
    if (len(shape) != 2 or any(isinstance(n, bool) or not isinstance(n, Integral) or n < 8 for n in shape)):
        raise ValueError('shape must contain two integer dimensions of at least eight')
    if (isinstance(anatomy_seed, bool) or not isinstance(anatomy_seed, Integral) or anatomy_seed < 0):
        raise ValueError('anatomy_seed must be a nonnegative integer')
    if (not np.isfinite(voxel_spacing_mm) or voxel_spacing_mm <= 0
            or not np.isfinite(focal_radius_mm) or focal_radius_mm <= 0
            or not np.isfinite(delta_adc)):
        raise ValueError('spacing/radius must be positive and all parameters finite')
    if focal_shape not in ('circle', 'ellipse'):
        raise ValueError('focal_shape must be circle or ellipse')
    rng = np.random.default_rng(anatomy_seed)
    y, x = np.indices(shape, dtype=float)
    cy = (shape[0] - 1) / 2 + rng.uniform(-0.04, 0.04) * shape[0]
    cx = (shape[1] - 1) / 2 + rng.uniform(-0.04, 0.04) * shape[1]
    angle = rng.uniform(-0.3, 0.3)
    xrot = (x - cx) * np.cos(angle) + (y - cy) * np.sin(angle)
    yrot = -(x - cx) * np.sin(angle) + (y - cy) * np.cos(angle)
    rx = shape[1] * rng.uniform(0.34, 0.42)
    ry = shape[0] * rng.uniform(0.34, 0.42)
    tissue = (xrot / rx) ** 2 + (yrot / ry) ** 2 <= 1
    inner_scale = rng.uniform(0.52, 0.68)
    inner = (xrot / (rx * inner_scale)) ** 2 + (yrot / (ry * inner_scale)) ** 2 <= 1
    baseline = np.where(inner, 0.0009, np.where(tissue, 0.0016, 0.0))
    amplitude = np.where(inner, 0.8, np.where(tissue, 1.0, 0.0))
    radius = focal_radius_mm / voxel_spacing_mm
    aspect = 1.0 if focal_shape == 'circle' else 1.5
    focal = (xrot / (radius * aspect)) ** 2 + (yrot / (radius / aspect)) ** 2 <= 1
    if not focal.any() or np.any(focal & ~tissue):
        raise ValueError('focal region must contain samples and fit entirely within tissue')
    perturbed = baseline.copy()
    perturbed[focal] += delta_adc
    if np.any(perturbed < 0):
        raise ValueError('focal perturbation produces negative diffusivity')
    return ToyPhantom(baseline, perturbed, amplitude, focal, tissue, float(voxel_spacing_mm))
