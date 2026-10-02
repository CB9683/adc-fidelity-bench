# Planned uncertainty and adaptive-stopping protocol

**Status: PLANNED / UNIMPLEMENTED, 2026-10-02.** The CPU conventional-baseline pilot is complete. It has not trained a neural model, calibrated an ADC-error bound, or evaluated adaptive stopping. This protocol extends [BENCHMARK_SPEC.md](BENCHMARK_SPEC.md); it does not establish a clinical acceptance threshold or an implemented coverage guarantee.

## Inputs, grouping, and local-fidelity target

Keep all slices, perturbations, noise realizations, and acquisition variants of each source anatomy in one training, validation, calibration, or test split. Fit estimator parameters and image-derived scale rules on training data; select their settings on validation data and freeze them before final calibration. Calibration and final test anatomies remain disjoint from both. No final test result selects a model, threshold, budget, or policy.

For anatomy `a`, registered variant `v`, and repetition budget `b`, define the primary local-fidelity error:

```text
Q[a,v,b] = max over x in true tissue ROI of |ADC_hat[a,v,b,x] - ADC_ref[a,v,x]|.
```

ADC units are `mm²/s`; the reference is the noiseless acquisition-resolution fit. The true tissue ROI includes every acquisition voxel with a positive fine-resolution tissue fraction, including partial volume. This maximum protects local errors that a whole-image mean can conceal. Record failure as `Q = +infinity` when the requested estimate is invalid or fitting fails; do not remove its denominator.

Reference maps and true tissue/focal masks are calibration targets and evaluation inputs only. Test-time estimation and stopping receive the observed repetition prefix, b-values, declared acquisition metadata, and estimator diagnostics. They receive no noiseless amplitude/ADC, true ROI, focal location, paired unperturbed reference, or unseen repetitions. Default stopping validity requires finite, identified, converged outputs throughout the acquisition grid; any later image-derived support rule must be independently frozen and must retain evaluation of omitted true tissue as a failure.

Define a strictly positive, finite scalar scale `s[a,v,b]` in `mm²/s` from the observed prefix, for example a repetition-resampling variability summary with a fixed positive floor. Its functional form, resampling seeds, floor, and observed-input normalization are selected on training/validation data. It cannot use test truth, generator SNR, true tissue masks, or noise/reference realizations unavailable from the declared acquisition. An unavailable/invalid scale makes that budget ineligible for stopping. The scalar is a heuristic until calibrated.

## Simultaneous calibration across budgets and variants

Freeze the finite budget set and variant-generation registry first. Calibration contributes one score per independent anatomy, rather than treating its correlated images as independent observations:

```text
S[a] = max over every registered variant v and budget b of Q[a,v,b] / s[a,v,b].
k = ceil((n_cal + 1) * (1 - alpha)).
q = k-th smallest anatomy score if k <= n_cal; otherwise q = +infinity.
B[a,v,b] = q * s[a,v,b].
```

An invalid estimate/scale contributes an infinite score. Preserve ties and infinite scores; do not substitute the largest finite score or an interpolated quantile. An infinite `q` produces no finite bound and no stopping. This conservative construction may defer every case when low-budget fitting frequently fails; report that outcome.

The proposed application of the split-conformal rank argument is to exchangeable **whole-anatomy score groups**, conditional on the frozen training/validation choices. With identical registered variant generation and score construction for calibration and a new anatomy, it supports a marginal simultaneous bound event across that anatomy's registered variants and budgets. Correlation within an anatomy is absorbed by the maximum; it does not increase `n_cal`. This is a protocol inference from [Angelopoulos and Bates' split-conformal construction](https://arxiv.org/abs/2107.07511), pending implementation and verification. It gives no conditional guarantee for each SNR/focal subgroup, no guarantee for an arbitrary fixed calibration dataset, and no coverage promise under distribution shift. Infinite bounds are vacuous; operational finite-bound coverage and failure rates must be reported separately.

Use exploratory `tau = [5e-5, 1e-4, 2e-4, 5e-4] mm²/s` and `alpha = [0.1, 0.05, 0.01]`. These are benchmark sweeps, not clinical acceptance criteria. Evaluate every prespecified pair without selecting the best pair on test data. Separate claims about multiple sweep entries require their own multiplicity treatment.

## Frozen adaptive policy and acquisition accounting

At each prespecified budget, stop at the first prefix with valid outputs, a valid scale, and finite `B <= tau`. Otherwise acquire the next prefix. A failure at a budget prevents stopping there; it cannot be hidden by averaging only valid voxels. Non-stoppers consume the maximum budget, and final failures remain error-risk failures. Record stopping status, reason, bound, per-b-value repetitions, total acquisitions, and acquisition order. Paired noise choices and nested prefix pools follow the canonical benchmark.

Simultaneous calibration avoids interpreting separately calibrated fixed-budget intervals as valid after repeated checks. Under the stated assumptions, the simultaneous bound event implies that a falsely safe stop is contained in its failure event. This is a marginal statement over whole anatomy groups. It does **not** assert `P(error > tau | stopped) <= alpha`; selective risk among stopped cases is a separate empirical endpoint.

## Test endpoints and matched-risk comparisons

Report anatomy-weighted results and explicit requested/valid/failed/stopped denominators:

- ADC bias, MAE/RMSE, maximum tissue error, and maximum focal-ROI error; stratify by SNR, focal size, signed contrast, budget, and registered regime.
- Simultaneous and budget-specific bound coverage, usable finite-bound coverage, interval width (`2B`), unbounded fraction, and estimator failures. Invalid outputs are misses for operational coverage, not successful infinite intervals.
- False confidence `stopped AND Q > tau`, both over all cases and among stopped cases; also report local focal false confidence using focal error. Include risk-versus-deferral curves and final delivered-estimate error risk, charging final failures as violations.
- Signed focal change error, recovery ratio only for prespecified positive `epsilon` and `|delta_ref| > epsilon`, opposite-sign recovery, attenuation, and spillover/non-focal error. Freeze the exploratory attenuation criterion on validation data; report confident stopping with unacceptable attenuation separately from ADC-error tolerance.
- Mean/median and distribution of acquisitions saved, including maximum-budget non-stoppers and failures. Acquisition counts support retrospective count savings, not prospective scanner-time savings.

For each tolerance/risk pair, choose a fixed-budget comparator and adaptive-policy settings on validation anatomies using identical available acquisitions, processing assumptions, and evaluation targets. Freeze these choices before final calibration/test. On the independent test split, compare savings only when measured risks are comparable for the same prespecified error event; report risk differences and uncertainty alongside savings. An all-case marginal false-stop rate cannot be matched to a comparator's conditional accepted-case risk. If no comparator achieves the target, report that result instead of choosing another budget on test data. Include the maximum-budget comparator and distinguish plain fixed-budget estimation from any comparator that also defers.

## Unfamiliar regimes and sample-size limits

Create a versioned held-out registry before final evaluation, with separate familiar and unfamiliar entries. Examples for future implementation are elongated/multifocal shapes absent from training/validation/calibration, different spatial resampling, correlated noise, and multicoil magnitude/reconstruction models with declared covariance and coil-combination assumptions. Keep unfamiliar calibration data out of the familiar calibration set. Report each shift separately as empirical robustness evidence; familiar-distribution calibration does not guarantee shifted coverage. Any later regime-specific calibration requires a new disjoint calibration/test design.

The minimum calibration anatomy counts for a potentially finite order statistic are 9, 19, and 99 for the three proposed alpha values; these are only non-vacuity thresholds, not adequate precision or realism. The current two calibration toy anatomies yield infinite quantiles for every proposed alpha, and two test anatomies cannot support a useful risk/savings claim. Plan independent anatomy counts for the target rare-event precision before running the study. With sufficient groups, report anatomy-cluster bootstrap intervals for savings and summary metrics, and appropriate anatomy-level intervals for prespecified failure-event proportions; preserve all variants within each resampled group. Small-sample intervals remain exploratory and cannot validate anatomy realism.

Implementation must verify grouping/leakage, order-statistic boundaries/ties/infinity, simultaneous budget checks, scale/input restrictions, invalid-fit deferral, maximum-budget charging, and reproducible reporting before any calibrated-stopping result is claimed.
