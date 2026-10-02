"""Quantitative ADC units and full focal-region coverage are explicit."""

import json

import numpy as np
import pytest

from adc_fidelity_bench.evaluation.metrics import adc_error_summary, focal_change_recovery


def test_adc_error_summary_uses_adc_units_and_signed_bias():
    reference = np.array([[0.001, 0.002], [0.003, 0.004]])
    estimate = reference + np.array([[0.0001, -0.0002], [0.0003, -0.0004]])

    summary = adc_error_summary(estimate, reference)

    assert summary["bias"] == pytest.approx(-0.00005)
    assert summary["mae"] == pytest.approx(0.00025)
    assert summary["rmse"] == pytest.approx(np.sqrt(7.5e-8))
    assert (summary["n_requested"], summary["n_valid"], summary["n_invalid"]) == (4, 4, 0)
    json.dumps(summary, allow_nan=False)


def test_adc_error_summary_reports_requested_roi_and_invalid_denominator():
    reference = np.array([[0.001, 0.002], [np.nan, 0.004]])
    estimate = np.array([[0.0011, np.nan], [np.inf, 0.0043]])
    mask = np.array([[True, True], [False, False]])

    summary = adc_error_summary(estimate, reference, mask)

    assert summary["bias"] == pytest.approx(0.0001)
    assert summary["mae"] == pytest.approx(0.0001)
    assert summary["rmse"] == pytest.approx(0.0001)
    assert (summary["n_requested"], summary["n_valid"], summary["n_invalid"]) == (2, 1, 1)


def test_adc_error_summary_does_not_return_metrics_when_all_estimates_are_invalid():
    summary = adc_error_summary([np.nan, np.inf], [0.001, 0.002])
    assert summary == {"bias": None, "mae": None, "rmse": None,
                       "n_requested": 2, "n_valid": 0, "n_invalid": 2}
    json.dumps(summary, allow_nan=False)


@pytest.mark.parametrize("estimate,reference,mask", [
    ([0.001], [[0.001]], None),
    ([0.001], [np.nan], None),
    ([0.001], [np.inf], None),
    ([0.001], [0.001], [False]),
    ([0.001], [0.001], [1]),
    ([0.001], [0.001], [[True]]),
    ([], [], None),
])
def test_adc_error_summary_rejects_undefined_reference_or_roi(estimate, reference, mask):
    with pytest.raises(ValueError):
        adc_error_summary(estimate, reference, mask)


@pytest.mark.parametrize("change", [0.0002, -0.0002])
def test_focal_recovery_preserves_positive_and_negative_change_sign(change):
    baseline = np.array([[0.001, 0.002], [0.003, 0.004]])
    mask = np.array([[True, True], [False, False]])
    reference = baseline + change * mask
    estimate = baseline + 0.75 * change * mask

    summary = focal_change_recovery(estimate, baseline, reference, baseline, mask)

    assert summary["delta_reference"] == pytest.approx(change)
    assert summary["delta_estimate"] == pytest.approx(0.75 * change)
    assert summary["change_error"] == pytest.approx(-0.25 * change)
    assert summary["recovery_ratio"] == pytest.approx(0.75)
    assert (summary["n_requested"], summary["n_valid"], summary["n_invalid"]) == (2, 2, 0)
    json.dumps(summary, allow_nan=False)


def test_opposite_estimated_focal_change_has_negative_recovery_ratio():
    summary = focal_change_recovery([0.0008], [0.001], [0.0012], [0.001], [True])
    assert summary["recovery_ratio"] == pytest.approx(-1.0)


def test_near_zero_reference_change_has_no_recovery_ratio():
    summary = focal_change_recovery([0.0011], [0.001], [0.001 + 1e-10], [0.001], [True])
    assert summary["delta_reference"] == pytest.approx(1e-10)
    assert summary["delta_estimate"] == pytest.approx(0.0001)
    assert summary["recovery_ratio"] is None


def test_focal_recovery_rejects_zero_epsilon_even_with_nonzero_reference_change():
    with pytest.raises(ValueError, match="Epsilon"):
        focal_change_recovery([0.0012], [0.001], [0.0012], [0.001], [True], epsilon=0)


@pytest.mark.parametrize("perturbed,baseline", [
    ([0.0012, np.nan], [0.001, 0.001]),
    ([0.0012, 0.0012], [0.001, np.inf]),
    ([np.nan, np.nan], [0.001, 0.001]),
])
def test_missing_focal_estimate_prevents_full_roi_recovery_claim(perturbed, baseline):
    summary = focal_change_recovery(perturbed, baseline, [0.0012, 0.0014],
                                   [0.001, 0.001], [True, True])
    expected_valid = int(np.count_nonzero(np.isfinite(perturbed) & np.isfinite(baseline)))
    assert summary["delta_reference"] == pytest.approx(0.0003)
    assert summary["delta_estimate"] is None
    assert summary["change_error"] is None
    assert summary["recovery_ratio"] is None
    assert (summary["n_requested"], summary["n_valid"], summary["n_invalid"]) == (
        2, expected_valid, 2 - expected_valid)
    json.dumps(summary, allow_nan=False)


def test_focal_recovery_ignores_invalid_values_outside_identical_roi():
    summary = focal_change_recovery([0.0012, np.nan], [0.001, np.nan],
                                   [0.0012, np.nan], [0.001, np.nan], [True, False])
    assert summary["recovery_ratio"] == pytest.approx(1.0)
    assert summary["n_valid"] == 1
    assert summary["n_invalid"] == 0


@pytest.mark.parametrize("perturbed,baseline,reference,unperturbed_reference,mask,epsilon", [
    ([[0.0012]], [0.001], [0.0012], [0.001], [True], 1e-8),
    ([0.0012], [0.001], [np.nan], [0.001], [True], 1e-8),
    ([0.0012], [0.001], [0.0012], [np.inf], [True], 1e-8),
    ([0.0012], [0.001], [0.0012], [0.001], [False], 1e-8),
    ([0.0012], [0.001], [0.0012], [0.001], [1], 1e-8),
    ([0.0012], [0.001], [0.0012], [0.001], [[True]], 1e-8),
    ([0.0012], [0.001], [0.0012], [0.001], [True], -1),
    ([0.0012], [0.001], [0.0012], [0.001], [True], np.nan),
])
def test_focal_recovery_rejects_undefined_inputs(perturbed, baseline, reference,
                                              unperturbed_reference, mask, epsilon):
    with pytest.raises(ValueError):
        focal_change_recovery(perturbed, baseline, reference, unperturbed_reference,
                              mask, epsilon)
