# CPU Simulation and Conventional Baseline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Deliver an executable, tested simulation pilot that measures ADC error and recovery of synthetic focal changes across noise and repetition budgets.

**Architecture:** Generate original geometric phantoms and noiseless diffusion signals at fine resolution, average spatial blocks of signals, then add complex Gaussian noise and form magnitude repetitions. Compare magnitude averaging/log-linear ADC fitting with repetition-level Rician maximum-likelihood fitting. Separate anatomy groups before deriving cases, evaluate paired focal/unperturbed references, and write reproducible tables/plots outside Git.

**Tech Stack:** Python 3.11+, NumPy, SciPy, Matplotlib, pytest; JSON configuration and CSV/JSON results. First verification uses installed Python 3.13 tools with recorded versions.

---

## Scope and reviewed choices

This plan advances the reviewed benchmark specification into a small CPU implementation. No patient data, BrainWeb download, neural training, calibrated stopping claim, or clinical assertion is involved. The generated phantoms and numerical parameters are synthetic, not biological estimates. Published-source denoising comparators, calibrated uncertainty/stopping, anatomical resources, and learned models remain subsequent milestones.

The isolated branch is `feature/simulation-baselines`. Temporary worktree location is operational state and must not enter published documentation.

Initial b-values are `[0,1000] s/mm^2`, SNR is fixed reference `S0/sigma`, and sigma describes each independent complex component. Reference `S0=1` defines amplitude units only. Repetition budgets are counted per b-value. Acquisition-resolution references use the same log-linear fit applied to noiseless spatially averaged signals. Evaluation truth/masks never enter an estimator.

## Contracts

- Fine-resolution signals: `(b_value, height, width)`.
- Noisy repetitions: `(b_value, repetition, height, width)`.
- ADC outputs and validity/convergence masks: `(height, width)`.
- `monoexponential_signal(b_values, adc, s0=1.0)` returns signals with b-value first.
- `mix_tissue_signals(fractions, tissue_adc, tissue_s0, b_values)` mixes tissue signals, with fractions `(tissue, height, width)`.
- `block_average(data, factor)` averages the final two spatial axes; both must be divisible by an integer factor.
- `add_complex_noise(signal, repetitions=..., sigma=..., rng=...)` returns complex repetitions; zero noise is allowed for known-answer tests.
- `make_toy_phantom(anatomy_seed=..., shape=..., voxel_spacing_mm=..., focal_radius_mm=..., delta_adc=..., focal_shape=...)` returns unperturbed/perturbed ADC, S0, focal/tissue masks, and spacing. Its baseline anatomy is unchanged across perturbation parameters.
- `fit_log_linear_adc(signals, b_values)` returns `ADCResult(adc, valid, s0, converged)`; invalid voxels are NaN/false and negative unconstrained estimates are retained.
- `fit_rician_adc(magnitude, b_values, sigma=..., maxiter=...)` fits S0 and nonnegative ADC from all available repetitions under known simulated sigma. Return explicit convergence/validity. No ground-truth S0 or ADC initialization is available to this method.
- `adc_error_summary(estimate, reference, mask=None)` reports bias/MAE/RMSE and requested/valid/invalid counts, with null metrics if every requested estimate is invalid.
- `focal_change_recovery(perturbed_estimate, unperturbed_estimate, perturbed_reference, unperturbed_reference, mask, epsilon=...)` reports signed reference/estimated change, error, ratio for `abs(delta_ref)>epsilon`, and invalid counts. A missing focal voxel prevents an apparently successful complete-region recovery claim.
- `grouped_split(anatomy_ids, counts, seed)` allocates sorted unique anatomy IDs reproducibly to train, validation, calibration, and test; `validate_split_integrity` rejects overlaps.

## Task 1: Simulation primitives — root

**Files:** `src/adc_fidelity_bench/simulation/{__init__,signals,noise,partial_volume,phantoms}.py`, `tests/test_simulation.py`.

1. Write known-answer tests for exponential units, zero-noise recovery inputs, shape/parameter validation, signal mixtures differing from averaged ADC, complex noise moments, deterministic RNG, and anatomy invariance across perturbations.
2. Run `PYTHONPATH=src python3 -m pytest tests/test_simulation.py -q`; verify the missing-feature failure before implementation.
3. Implement the smallest numerical functions. Reject nonfinite/unphysical generator parameters and ill-defined spatial resampling.
4. Rerun the focused tests and inspect generated phantom geometry.

## Task 2: Conventional ADC estimators — independent agent

**Files:** `src/adc_fidelity_bench/baselines/{__init__,adc}.py`, `tests/test_baselines.py`.

1. Write and run failing tests for exact ADC units, arbitrary signal amplitude, negative estimates, invalid signals, b-value validation, known-noise Rician fitting, constraints, and convergence failure reporting.
2. Implement log-linear fitting and stable Rician likelihood using scaled Bessel functions. Sigma is known simulation metadata; label that comparator assumption in reports.
3. Run `PYTHONPATH=src python3 -m pytest tests/test_baselines.py -q` and inspect convergence tests.

## Task 3: Split integrity and quantitative metrics — independent agent

**Files:** `src/adc_fidelity_bench/data/{__init__,splits}.py`, `src/adc_fidelity_bench/evaluation/{__init__,metrics}.py`, `tests/test_splits.py`, `tests/test_metrics.py`.

1. Write and run failing tests for group isolation, deterministic assignments, duplicate IDs, counts, and derived-example membership.
2. Test signed/negative focal changes, near-zero truth, reference masks, invalid estimates, and explicit denominators.
3. Implement minimal split and metric functions and rerun the focused tests.

## Task 4: Executable CPU pilot and reporting — root after integration

**Files:** `src/adc_fidelity_bench/{cli,__main__,pipeline}.py`, `configs/pilot.json`, `tests/test_pipeline.py`, `pyproject.toml`, `requirements-lock.txt`.

1. Fix small synthetic parameter ranges and entire-anatomy splits in a complete executable config; distinguish it from the draft example.
2. Write failing integration tests for deterministic CSV results, budget accounting, configuration errors, output preservation, and absence of truth inputs to estimators.
3. Implement `python -m adc_fidelity_bench benchmark --config configs/pilot.json --output outputs/pilot`.
4. Preserve maximum-budget repetition pools and use nested prefixes for budgets. Record whether paired cases use common or independent noise; no reuse across train/test anatomy groups.
5. Save config, synthetic split IDs, seeds, software versions, per-case errors/recovery/convergence, and anatomy-aggregated tables. Make budget and focal-recovery plots with explicit units and toy-pilot limitations.
6. Add a minimal reproducible dependency specification and installation instructions. Do not install into global Python.

## Task 5: Verify, review, and integrate

**Files:** `README.md`, `tests/README.md`, `docs/BENCHMARK_SPEC.md`, `docs/PILOT_RESULTS.md` after actual results.

1. Run the entire meaningful numerical/integration test suite.
2. Run a small CPU benchmark and verify report tables, convergence/failure counts, and plot rendering. Record actual measured values without clinical extrapolation.
3. Obtain independent specification and numerical/code-quality review; fix material findings and rerun affected checks.
4. Review complete changed content and history for privacy before publication; keep generated arrays and unreviewed outputs outside Git.
5. Integrate the verified branch into the main workspace and publish the implementation through the already authorized repository workflow.
6. Continue next with published denoising comparisons and explicit uncertainty/stopping protocol. Large-scale training remains gated on that quantitative protocol review.
