"""Matched spatial/pointwise models and known-answer paired training losses."""

import importlib
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')


def module():
    import adc_fidelity_bench.learning as learning
    assert (Path(learning.__file__).parent / 'ablation_model.py').is_file(), 'Ablation models are not implemented'
    return importlib.import_module('adc_fidelity_bench.learning.ablation_model')


def loss_inputs():
    healthy = torch.tensor([[[[0.001, 0.002]]], [[[0.003, 0.004]]]], dtype=torch.float64,
                           requires_grad=True)
    prediction = torch.tensor([[[[0.002, 0.004]]], [[[0.006, 0.008]]]], dtype=torch.float64,
                              requires_grad=True)
    healthy_target = torch.zeros_like(healthy)
    target = torch.zeros_like(prediction)
    tissue = torch.tensor([[[[True, False]]], [[[True, True]]]])
    focal = tissue.clone()
    return [healthy, prediction, healthy_target, target, tissue, focal]


def test_spatial_factory_reuses_original_model_and_exact_copied_state():
    from adc_fidelity_bench.learning.model import SmallADCCNN
    torch.manual_seed(23)
    original = SmallADCCNN()
    spatial = module().make_ablation_model('spatial')
    assert type(spatial) is SmallADCCNN
    spatial.load_state_dict(original.state_dict(), strict=True)
    features = torch.rand(3, 2, 5, 7)
    torch.testing.assert_close(spatial(features), original(features), rtol=0, atol=0)
    assert sum(p.numel() for p in spatial.parameters()) == 2641


def test_pointwise_model_retains_shape_and_matched_capacity():
    model = module().make_ablation_model('pointwise')
    prediction = model(torch.ones(3, 2, 5, 7))
    assert prediction.shape == (3, 1, 5, 7)
    assert torch.isfinite(prediction).all() and (prediction > 0).all()
    assert sum(p.numel() for p in model.parameters()) == 2647


def test_pointwise_model_has_no_learned_neighbour_dependence():
    model = module().make_ablation_model('pointwise')
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.02)
    features = torch.ones(1, 2, 5, 7)
    changed = features.clone()
    changed[:, :, 2, 3] += 1
    difference = model(changed) - model(features)
    assert difference[0, 0, 2, 3] > 0
    difference[0, 0, 2, 3] = 0
    torch.testing.assert_close(difference, torch.zeros_like(difference), rtol=0, atol=0)


@pytest.mark.parametrize('architecture', ['spatial', 'pointwise'])
def test_zero_logits_return_softplus_in_physical_adc_units(architecture):
    model = module().make_ablation_model(architecture)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    np.testing.assert_allclose(model(torch.ones(1, 2, 3, 4)).detach().numpy(),
                               np.log(2) * 1e-3, rtol=1e-6)


@pytest.mark.parametrize('architecture', ['missing', '', None, 7, ['spatial']])
def test_unknown_architectures_are_rejected(architecture):
    with pytest.raises(ValueError, match='architecture'):
        module().make_ablation_model(architecture)


@pytest.mark.parametrize('shape', [(2, 3, 4), (1, 3, 3, 4)])
def test_pointwise_rejects_missing_batch_or_wrong_channels(shape):
    with pytest.raises(ValueError, match='shape'):
        module().make_ablation_model('pointwise')(torch.ones(shape))


@pytest.mark.parametrize('architecture', ['spatial', 'pointwise'])
def test_restricted_state_reload_preserves_prediction(architecture, tmp_path):
    model = module().make_ablation_model(architecture)
    features = torch.rand(2, 2, 4, 5)
    expected = model(features).detach()
    checkpoint = tmp_path / 'state.pt'
    torch.save(model.state_dict(), checkpoint)
    restored = module().make_ablation_model(architecture)
    restored.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    torch.testing.assert_close(restored(features), expected, rtol=0, atol=0)


def test_paired_loss_weights_each_image_equally_and_matches_original_mse():
    from adc_fidelity_bench.learning.model import masked_scaled_mse
    values = loss_inputs()
    result = module().paired_loss_components(*values)
    assert set(result) == {'mse', 'change_mse'}
    assert all(value.ndim == 0 for value in result.values())
    assert result['mse'].item() == pytest.approx(16.875)
    assert result['change_mse'].item() == pytest.approx(6.625)
    healthy, prediction, healthy_target, target, tissue, _ = values
    original = masked_scaled_mse(torch.cat([healthy, prediction]),
                                 torch.cat([healthy_target, target]), torch.cat([tissue, tissue]))
    torch.testing.assert_close(result['mse'], original)


def test_paired_loss_known_answer_gradients_keep_equal_pair_weight():
    values = loss_inputs()
    result = module().paired_loss_components(*values)
    (result['mse'] + result['change_mse']).backward()
    torch.testing.assert_close(values[0].grad, torch.tensor([[[[-500., 0.]]], [[[-1000., -750.]]]],
                                                           dtype=torch.float64))
    torch.testing.assert_close(values[1].grad, torch.tensor([[[[2000., 0.]]], [[[3250., 3750.]]]],
                                                           dtype=torch.float64))


@pytest.mark.parametrize('delta', [-0.0003, 0.0, 0.0003])
def test_exact_signed_and_null_pair_targets_have_zero_loss(delta):
    healthy_target = torch.full((2, 1, 2, 3), 0.001, dtype=torch.float64)
    target = healthy_target + delta
    mask = torch.ones_like(target, dtype=torch.bool)
    result = module().paired_loss_components(healthy_target, target, healthy_target, target, mask, mask)
    assert result['mse'].item() == 0 and result['change_mse'].item() == 0


def test_common_adc_bias_cancels_only_in_signed_change_component():
    healthy_target = torch.full((1, 1, 2, 2), 0.001, dtype=torch.float64)
    target = healthy_target - 0.0003
    mask = torch.ones_like(target, dtype=torch.bool)
    result = module().paired_loss_components(healthy_target + 0.0001, target + 0.0001,
                                             healthy_target, target, mask, mask)
    assert result['mse'].item() == pytest.approx(0.01)
    assert result['change_mse'].item() == pytest.approx(0, abs=1e-25)


@pytest.mark.parametrize('delta', [-0.0003, 0.0003])
def test_opposite_signed_change_is_penalized(delta):
    healthy = torch.full((1, 1, 1, 2), 0.001, dtype=torch.float64)
    mask = torch.ones_like(healthy, dtype=torch.bool)
    result = module().paired_loss_components(healthy, healthy - delta, healthy, healthy + delta, mask, mask)
    assert result['change_mse'].item() == pytest.approx(0.36)


def test_null_pair_false_change_contributes_to_change_loss():
    healthy = torch.full((1, 1, 2, 2), 0.001, dtype=torch.float64)
    mask = torch.ones_like(healthy, dtype=torch.bool)
    result = module().paired_loss_components(healthy, healthy + 0.0002, healthy, healthy, mask, mask)
    assert result['change_mse'].item() == pytest.approx(0.04)


def test_nan_values_outside_tissue_never_enter_loss_or_gradients():
    arrays = [torch.tensor([[[[value, float('nan')]]]], dtype=torch.float64, requires_grad=True)
              for value in (0.001, 0.002, 0.001, 0.001)]
    mask = torch.tensor([[[[True, False]]]])
    result = module().paired_loss_components(*arrays, mask, mask)
    assert result['mse'].item() == pytest.approx(0.5)
    assert result['change_mse'].item() == pytest.approx(1)
    (result['mse'] + result['change_mse']).backward()
    for array in arrays:
        assert torch.isfinite(array.grad).all()
        assert array.grad[0, 0, 0, 1] == 0
    torch.testing.assert_close(arrays[0].grad, torch.tensor([[[[-2000., 0.]]]], dtype=torch.float64))
    torch.testing.assert_close(arrays[1].grad, torch.tensor([[[[3000., 0.]]]], dtype=torch.float64))


@pytest.mark.parametrize('field', range(4))
@pytest.mark.parametrize('invalid', [float('nan'), float('inf')])
def test_nonfinite_requested_adcs_are_rejected(field, invalid):
    values = loss_inputs()
    values[field] = values[field].detach().clone()
    values[field][0, 0, 0, 0] = invalid
    with pytest.raises(ValueError, match='finite'):
        module().paired_loss_components(*values)


def test_finite_adc_values_that_overflow_scaled_loss_are_rejected():
    healthy = torch.full((1, 1, 1, 1), 1e30)
    prediction = -healthy
    target = torch.zeros_like(healthy)
    mask = torch.ones_like(healthy, dtype=torch.bool)
    with pytest.raises(ValueError, match='finite'):
        module().paired_loss_components(healthy, prediction, target, target, mask, mask)


@pytest.mark.parametrize('field', range(6))
def test_every_argument_must_be_a_matching_tensor(field):
    values = loss_inputs()
    values[field] = values[field].detach().numpy()
    with pytest.raises(ValueError, match='tensor'):
        module().paired_loss_components(*values)


@pytest.mark.parametrize('invalid', ['target_shape', 'channels', 'empty_batch', 'empty_spatial',
                                    'tissue_dtype', 'focal_dtype', 'empty_tissue_image',
                                    'empty_focal_image', 'focal_outside_tissue', 'complex', 'device'])
def test_invalid_masks_shapes_and_adc_types_are_rejected(invalid):
    values = loss_inputs()
    if invalid == 'target_shape':
        values[3] = values[3][:1]
    elif invalid == 'channels':
        values = [value.expand(2, 2, 1, 2) for value in values]
    elif invalid == 'empty_batch':
        values = [value[:0] for value in values]
    elif invalid == 'empty_spatial':
        values = [value[..., :0] for value in values]
    elif invalid == 'tissue_dtype':
        values[4] = values[4].float()
    elif invalid == 'focal_dtype':
        values[5] = values[5].long()
    elif invalid == 'empty_tissue_image':
        values[4][1] = False
    elif invalid == 'empty_focal_image':
        values[5][1] = False
    elif invalid == 'focal_outside_tissue':
        values[5][0, 0, 0, 1] = True
    elif invalid == 'complex':
        values[1] = values[1].to(torch.complex128)
    else:
        values[3] = torch.zeros(values[3].shape, device='meta')
    with pytest.raises(ValueError):
        module().paired_loss_components(*values)
