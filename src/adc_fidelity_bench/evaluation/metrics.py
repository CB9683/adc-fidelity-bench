"""ADC evaluation with explicit requested and invalid voxel counts.

Reference arrays and masks belong to evaluation. They are never estimator inputs.
ADC differences retain the input units, normally mm^2/s; recovery ratios are
dimensionless.
"""

import numpy as np
from numpy.typing import ArrayLike


def _roi(shape: tuple[int, ...], mask: ArrayLike | None) -> np.ndarray:
    if mask is None:
        roi = np.ones(shape, dtype=bool)
    else:
        roi = np.asarray(mask)
        if roi.dtype != np.dtype(bool) or roi.shape != shape:
            raise ValueError("Mask must be boolean and have exactly the ADC array shape.")
    if not np.any(roi):
        raise ValueError("The requested ROI must contain at least one voxel.")
    return roi


def _matching_arrays(*arrays: ArrayLike) -> tuple[np.ndarray, ...]:
    converted = tuple(np.asarray(array, dtype=float) for array in arrays)
    if any(array.shape != converted[0].shape for array in converted[1:]):
        raise ValueError("All ADC arrays must have exactly the same shape.")
    return converted


def _check_reference(reference: np.ndarray, roi: np.ndarray) -> None:
    if not np.all(np.isfinite(reference[roi])):
        raise ValueError("Reference ADC must be finite throughout the requested ROI.")


def _counts(roi: np.ndarray, valid: np.ndarray) -> dict[str, int]:
    n_requested = int(np.count_nonzero(roi))
    n_valid = int(np.count_nonzero(valid))
    return {"n_requested": n_requested, "n_valid": n_valid,
            "n_invalid": n_requested - n_valid}


def adc_error_summary(
    estimate: ArrayLike, reference: ArrayLike, mask: ArrayLike | None = None
) -> dict[str, float | int | None]:
    """Report errors over finite estimates and expose their requested denominator.

    Bias is mean(estimate - reference). Invalid estimates remain in n_requested
    and n_invalid, while the three numerical errors use n_valid. No numerical
    errors are reported when the requested region contains no valid estimate.
    """
    estimate_array, reference_array = _matching_arrays(estimate, reference)
    roi = _roi(reference_array.shape, mask)
    _check_reference(reference_array, roi)
    valid = roi & np.isfinite(estimate_array)
    summary = {"bias": None, "mae": None, "rmse": None, **_counts(roi, valid)}
    if summary["n_valid"]:
        errors = estimate_array[valid] - reference_array[valid]
        summary.update(bias=float(np.mean(errors)), mae=float(np.mean(np.abs(errors))),
                       rmse=float(np.sqrt(np.mean(np.square(errors)))))
    return summary


def focal_change_recovery(
    perturbed_estimate: ArrayLike,
    unperturbed_estimate: ArrayLike,
    perturbed_reference: ArrayLike,
    unperturbed_reference: ArrayLike,
    mask: ArrayLike,
    epsilon: float = 1e-8,
) -> dict[str, float | int | None]:
    """Measure a signed paired ADC change over the entire fixed focal ROI.

    Change error is delta_estimate - delta_reference. A recovery ratio is
    reported for positive or negative reference changes above epsilon in
    absolute magnitude. Any invalid paired estimate suppresses estimated
    full-ROI metrics, preventing missing focal voxels from appearing recovered.
    """
    if mask is None:
        raise ValueError("Focal-change evaluation requires an explicit boolean ROI.")
    try:
        epsilon = float(epsilon)
    except (TypeError, ValueError) as error:
        raise ValueError("Epsilon must be a finite positive scalar.") from error
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Epsilon must be a finite positive scalar.")

    perturbed, unperturbed, perturbed_ref, unperturbed_ref = _matching_arrays(
        perturbed_estimate, unperturbed_estimate, perturbed_reference, unperturbed_reference)
    roi = _roi(perturbed_ref.shape, mask)
    _check_reference(perturbed_ref, roi)
    _check_reference(unperturbed_ref, roi)
    valid = roi & np.isfinite(perturbed) & np.isfinite(unperturbed)
    delta_reference = float(np.mean(perturbed_ref[roi] - unperturbed_ref[roi]))
    summary = {"delta_reference": delta_reference, "delta_estimate": None,
               "change_error": None, "recovery_ratio": None, **_counts(roi, valid)}
    if summary["n_invalid"] == 0:
        delta_estimate = float(np.mean(perturbed[roi] - unperturbed[roi]))
        summary.update(delta_estimate=delta_estimate,
                       change_error=delta_estimate - delta_reference)
        if abs(delta_reference) > epsilon:
            summary["recovery_ratio"] = delta_estimate / delta_reference
    return summary
