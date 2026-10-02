"""Observed-input CNN, physical units, and tissue-only loss contracts."""

import importlib
import inspect
from pathlib import Path

import numpy as np
import pytest

torch = None


def _model_module():
    assert (Path(__file__).parents[1] / "src/adc_fidelity_bench/learning/model.py").is_file(), "The observed-input CNN module is not implemented yet."
    global torch
    torch = pytest.importorskip("torch")
    return importlib.import_module("adc_fidelity_bench.learning.model")


def test_observed_features_use_mean_repetitions_and_observed_b0_percentile():
    b0 = np.array([[[2, 4], [6, 8]], [[4, 6], [8, 10]]], dtype=float)
    magnitude = np.stack((b0, b0 * 0.5))
    features = _model_module().observed_features(magnitude)
    means = magnitude.mean(axis=1)
    expected = means / np.percentile(means[0], 95)
    assert features.shape == (2, 2, 2)
    assert features.dtype == np.float32
    np.testing.assert_allclose(features, expected, rtol=1e-6)


def test_observed_features_use_only_repetitions_supplied_by_caller():
    magnitude = np.array([[[[1]], [[100]]], [[[0.5]], [[100]]]])
    features = _model_module().observed_features(magnitude[:, :1])
    np.testing.assert_array_equal(features, [[[1]], [[0.5]]])
    assert not np.array_equal(features, _model_module().observed_features(magnitude))


def test_observed_features_apply_scale_floor_without_reference_inputs():
    module = _model_module()
    magnitude = np.zeros((2, 2, 2, 3))
    magnitude[1] = 1e-6
    np.testing.assert_array_equal(module.observed_features(magnitude)[1], np.ones((2, 3)))
    assert list(inspect.signature(module.observed_features).parameters) == ["magnitude"]
    assert list(inspect.signature(module.predict_adc).parameters) == ["model", "magnitude", "device"]


def test_observed_features_preserve_finite_large_signal_amplitude_units():
    features = _model_module().observed_features(np.full((2, 2, 2, 2), 1e308))
    np.testing.assert_array_equal(features, np.ones((2, 2, 2), dtype=np.float32))


@pytest.mark.parametrize("magnitude", [
    np.ones((2, 2, 3)), np.ones((3, 2, 2, 3)), np.empty((2, 0, 2, 3)),
    np.empty((2, 2, 0, 3)), np.empty((2, 2, 2, 0)),
    np.full((2, 1, 2, 2), -1), np.full((2, 1, 2, 2), np.nan),
    np.full((2, 1, 2, 2), np.inf), np.full((2, 1, 2, 2), 1j),
])
def test_observed_features_reject_invalid_signal_values_and_axes(magnitude):
    with pytest.raises(ValueError):
        _model_module().observed_features(magnitude)


def test_cnn_retains_spatial_shape_and_outputs_positive_adc_units():
    module = _model_module()
    torch.manual_seed(11)
    model = module.SmallADCCNN()
    prediction = model(torch.ones((3, 2, 5, 7)))
    assert prediction.shape == (3, 1, 5, 7)
    assert torch.all(torch.isfinite(prediction))
    assert torch.all(prediction > 0)
    assert torch.all(prediction < 0.01)
    assert sum(parameter.numel() for parameter in model.parameters()) == 2641


def test_cnn_zero_logits_return_softplus_in_physical_adc_units():
    module = _model_module()
    model = module.SmallADCCNN()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    prediction = model(torch.ones((1, 2, 3, 4)))
    np.testing.assert_allclose(prediction.detach().numpy(), np.log(2) * 1e-3, rtol=1e-6)


@pytest.mark.parametrize("shape", [(2, 4, 4), (1, 3, 4, 4)])
def test_cnn_rejects_missing_batch_and_wrong_channel_axes(shape):
    with pytest.raises(ValueError):
        _model_module().SmallADCCNN()(torch.ones(shape))


def test_masked_loss_means_each_image_before_averaging_images():
    module = _model_module()
    prediction = torch.tensor([[[[0.001, 0.002]]], [[[0.003, 0.004]]]], requires_grad=True)
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[[[True, False]]], [[[True, True]]]])
    loss = module.masked_scaled_mse(prediction, target, mask)
    # Per-image losses are 1 and (9+16)/2, so each image weighs equally.
    assert loss.item() == pytest.approx(6.75)
    loss.backward()
    assert torch.all(torch.isfinite(prediction.grad))
    assert prediction.grad[0, 0, 0, 1] == 0


def test_masked_loss_selects_requested_voxels_before_nan_arithmetic():
    module = _model_module()
    prediction = torch.tensor([[[[0.002, float("nan")]]]], requires_grad=True)
    target = torch.tensor([[[[0.001, float("nan")]]]])
    mask = torch.tensor([[[[True, False]]]])
    loss = module.masked_scaled_mse(prediction, target, mask)
    assert loss.item() == pytest.approx(1)
    loss.backward()
    np.testing.assert_allclose(prediction.grad.numpy(), [[[[2000, 0]]]], rtol=1e-6)


@pytest.mark.parametrize("invalid_field", ["prediction", "target"])
def test_masked_loss_rejects_nonfinite_requested_voxels(invalid_field):
    module = _model_module()
    prediction = torch.ones((1, 1, 2, 2))
    target = torch.ones_like(prediction)
    if invalid_field == "prediction":
        prediction[0, 0, 0, 0] = float("nan")
    else:
        target[0, 0, 0, 0] = float("inf")
    with pytest.raises(ValueError):
        module.masked_scaled_mse(prediction, target, torch.ones_like(prediction, dtype=torch.bool))


@pytest.mark.parametrize("invalid_case", ["target_shape", "mask_shape", "mask_dtype", "empty_image", "channel_axis"])
def test_masked_loss_rejects_invalid_shapes_and_empty_image_masks(invalid_case):
    module = _model_module()
    prediction = torch.ones((2, 1, 2, 2))
    target = torch.zeros_like(prediction)
    mask = torch.ones_like(prediction, dtype=torch.bool)
    if invalid_case == "target_shape":
        target = target[:1]
    elif invalid_case == "mask_shape":
        mask = mask[:1]
    elif invalid_case == "mask_dtype":
        mask = mask.float()
    elif invalid_case == "empty_image":
        mask[0] = False
    else:
        prediction = prediction.expand((2, 2, 2, 2))
        target = torch.zeros_like(prediction)
        mask = torch.ones_like(prediction, dtype=torch.bool)
    with pytest.raises(ValueError):
        module.masked_scaled_mse(prediction, target, mask)


def test_cpu_state_dict_reload_preserves_exact_prediction(tmp_path):
    module = _model_module()
    torch.manual_seed(11)
    model = module.SmallADCCNN()
    model.train()
    magnitude = np.random.default_rng(12).uniform(0.1, 1, size=(2, 4, 4, 5))
    prediction = module.predict_adc(model, magnitude)
    assert prediction.shape == (4, 5)
    assert model.training
    checkpoint = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = module.SmallADCCNN()
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    np.testing.assert_array_equal(module.predict_adc(restored, magnitude), prediction)


def test_predict_adc_disables_gradients_and_uses_eval_mode():
    module = _model_module()
    class RecordingModel(torch.nn.Module):
        def forward(self, features):
            assert not self.training
            assert not torch.is_grad_enabled()
            return features[:, :1] * 0.001

    model = RecordingModel()
    model.train()
    prediction = module.predict_adc(model, np.ones((2, 1, 3, 4)))
    assert model.training
    np.testing.assert_array_equal(prediction, np.full((3, 4), 0.001, dtype=np.float32))
