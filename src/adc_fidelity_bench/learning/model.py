"""Small observed-input ADC CNN and equal-image tissue training loss."""

import numpy as np
import torch
from torch import nn


def observed_features(magnitude):
    """Normalize observed (2, R, H, W) magnitude means into float32 channels.

    The scale is the 95th percentile of the observed mean b0 image, floored
    at 1e-6 in magnitude units. Only supplied repetitions are used; reference
    ADC, tissue masks, and generator noise metadata are unavailable here.
    """
    if np.iscomplexobj(magnitude):
        raise ValueError("magnitude must be real-valued")
    try:
        observation = np.asarray(magnitude, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("magnitude must be a numeric array") from exc
    if (observation.ndim != 4 or observation.shape[0] != 2
            or any(size < 1 for size in observation.shape[1:])):
        raise ValueError("magnitude must have nonempty shape (2, R, H, W)")
    if not np.all(np.isfinite(observation)) or np.any(observation < 0):
        raise ValueError("magnitude must be finite and nonnegative")
    # Divide before summing so finite magnitude units cannot overflow a sum.
    means = (observation / observation.shape[1]).sum(axis=1)
    scale = max(float(np.percentile(means[0], 95)), 1e-6)
    with np.errstate(over="ignore", invalid="ignore"):
        features = (means / scale).astype(np.float32)
    if not np.all(np.isfinite(features)):
        raise ValueError("normalized observations must be representable as finite float32")
    return features


class SmallADCCNN(nn.Module):
    """2,641-parameter, 5×5-context CNN returning ADC in mm²/s."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1), nn.Softplus(),
        )

    def forward(self, features):
        if features.ndim != 4 or features.shape[1] != 2:
            raise ValueError("CNN features must have shape (N, 2, H, W)")
        return self.layers(features) * 1e-3


def predict_adc(model, magnitude, device="cpu"):
    """Predict an (H, W) ADC array from observed magnitudes only."""
    features = torch.as_tensor(observed_features(magnitude)[None], device=device)
    previous_mode = model.training
    model.to(device)
    model.eval()
    try:
        with torch.no_grad():
            prediction = model(features)
        return prediction[0, 0].detach().cpu().numpy()
    finally:
        model.train(previous_mode)


def masked_scaled_mse(prediction, target, mask):
    """Mean per-image tissue MSE after scaling ADC values by 1e3.

    Boolean selection occurs before subtraction so NaNs outside tissue never
    enter the loss or its gradients. Every image must request at least one
    voxel, and all requested prediction/target values must be finite.
    """
    if not all(isinstance(value, torch.Tensor) for value in (prediction, target, mask)):
        raise ValueError("prediction, target, and mask must be tensors")
    if (prediction.ndim != 4 or prediction.shape[1] != 1
            or any(size < 1 for size in prediction.shape)
            or target.shape != prediction.shape or mask.shape != prediction.shape):
        raise ValueError("prediction, target, and mask must have identical nonempty shape (N, 1, H, W)")
    if mask.dtype != torch.bool:
        raise ValueError("mask must have boolean dtype")
    if torch.is_complex(prediction) or torch.is_complex(target):
        raise ValueError("ADC prediction and target must be real-valued")
    losses = []
    for image in range(prediction.shape[0]):
        requested = mask[image]
        if not torch.any(requested):
            raise ValueError("every image must request at least one tissue voxel")
        estimate = prediction[image][requested]
        reference = target[image][requested]
        if not torch.all(torch.isfinite(estimate)) or not torch.all(torch.isfinite(reference)):
            raise ValueError("requested ADC values must be finite")
        losses.append(((estimate - reference) * 1e3).square().mean())
    return torch.stack(losses).mean()
