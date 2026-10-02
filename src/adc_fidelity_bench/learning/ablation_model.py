"""Matched-context ADC networks and paired supervised loss components."""

import torch
from torch import nn

from .model import SmallADCCNN


class PointwiseADCCNN(nn.Module):
    """2,647-parameter ADC model with no learned spatial mixing.

    The shared observed-input preprocessing still uses an image-wide b0
    percentile. This architecture isolates learned context, not all spatial
    dependence in the inference pipeline.
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(2, 49, kernel_size=1), nn.ReLU(),
            nn.Conv2d(49, 49, kernel_size=1), nn.ReLU(),
            nn.Conv2d(49, 1, kernel_size=1), nn.Softplus(),
        )

    def forward(self, features):
        if features.ndim != 4 or features.shape[1] != 2:
            raise ValueError('CNN features must have shape (N, 2, H, W)')
        return self.layers(features) * 1e-3


def make_ablation_model(architecture):
    """Construct the original spatial CNN or its matched-capacity control."""
    if not isinstance(architecture, str) or architecture not in ('spatial', 'pointwise'):
        raise ValueError('architecture must be spatial or pointwise')
    return SmallADCCNN() if architecture == 'spatial' else PointwiseADCCNN()


def paired_loss_components(healthy_prediction, prediction, healthy_target, target,
                           tissue_mask, focal_mask):
    """Return equal-image tissue MSE and equal-pair signed focal-change MSE.

    ADC differences are scaled by 1000 before squaring. Every pair contributes
    to the change component, including zero-change references. True masks and
    paired targets support supervision only; the models receive observations.
    Values outside requested tissue are selected out before subtraction or
    multiplication, so NaNs there cannot contaminate losses or gradients.
    """
    adcs = (healthy_prediction, prediction, healthy_target, target)
    tensors = (*adcs, tissue_mask, focal_mask)
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        raise ValueError('All paired-loss arguments must be tensors')
    shape = prediction.shape
    if (prediction.ndim != 4 or shape[1] != 1 or any(size < 1 for size in shape)
            or any(value.shape != shape for value in tensors)):
        raise ValueError('All paired-loss tensors must have identical nonempty shape (N, 1, H, W)')
    if any(value.device != prediction.device for value in tensors):
        raise ValueError('All paired-loss tensors must be on the same device')
    if tissue_mask.dtype != torch.bool or focal_mask.dtype != torch.bool:
        raise ValueError('Tissue and focal masks must have boolean dtype')
    if any(torch.is_complex(value) or value.dtype == torch.bool for value in adcs):
        raise ValueError('ADC predictions and targets must be real-valued numeric tensors')
    axes = (1, 2, 3)
    tissue_counts = tissue_mask.sum(dim=axes)
    focal_counts = focal_mask.sum(dim=axes)
    if torch.any(tissue_counts == 0) or torch.any(focal_counts == 0):
        raise ValueError('Every image must request nonempty tissue and focal masks')
    if torch.any(focal_mask & ~tissue_mask):
        raise ValueError('Every focal mask must be a subset of its tissue mask')
    if any(not torch.isfinite(value[tissue_mask]).all() for value in adcs):
        raise ValueError('Requested ADC predictions and targets must be finite')

    # Select before arithmetic. In particular, NaN * 0 would be unsafe.
    tissue_values = [torch.where(tissue_mask, value, 0) for value in adcs]
    healthy_errors = (tissue_values[0] - tissue_values[2]) * 1e3
    errors = (tissue_values[1] - tissue_values[3]) * 1e3
    healthy_mse = healthy_errors.square().sum(dim=axes) / tissue_counts
    perturbed_mse = errors.square().sum(dim=axes) / tissue_counts
    mse = torch.cat((healthy_mse, perturbed_mse)).mean()

    focal_values = [torch.where(focal_mask, value, 0) for value in adcs]
    estimated_change = (focal_values[1] - focal_values[0]).sum(dim=axes) / focal_counts
    reference_change = (focal_values[3] - focal_values[2]).sum(dim=axes) / focal_counts
    change_mse = ((estimated_change - reference_change) * 1e3).square().mean()
    if not torch.isfinite(mse) or not torch.isfinite(change_mse):
        raise ValueError('Scaled paired-loss components must be finite')
    return {'mse': mse, 'change_mse': change_mse}
