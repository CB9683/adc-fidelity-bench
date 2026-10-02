# Controlled simulation and quantitative evaluation

Status: proposed for review; no experiments have run. The example configuration is explicitly non-executable until missing parameters are fixed.

## Simulation and reference

1. Assign fine-resolution tissue fractions, `S0`, and `D`; create paired unperturbed and focal-abnormality cases. Record geometry/size and ADC contrast in physical units.
2. Generate tissue signals `S_t(b) = S0_t exp(-b D_t)`. Mix tissue signals by fractions, then apply the documented acquisition point-spread/resampling operator to signals, including boundary handling.
3. Fit the acquisition-resolution reference ADC from noiseless acquisition-resolution signals. Mixtures need not be monoexponential; this reference depends on b-values and fit. Keep assigned fine-resolution parameters separately.
4. Initially add independent Gaussian real/imaginary noise with standard deviation `sigma`, then form magnitude. Document this single-coil image-domain approximation; multicoil/k-space extensions need explicit reconstruction and covariance assumptions.
5. Define reference SNR as fixed generator-reference `S0 / sigma` for one complex component. The example's reference `S0=1` is an amplitude convention for generation/noise specification; tissue amplitudes may differ. Estimator normalization is separately defined using observed inputs, never per-voxel division by noiseless truth. Preserve individual repetitions. Do not substitute Gaussian noise on an ADC map.

The initial isotropic model does not represent all tissue diffusion or scanner artifacts. Distinguish magnitude averaging from complex averaging; comparators must follow their declared processing order.

## Reversible pilot settings

Proposed illustrative b-values: `[0, 1000] s/mm^2`; repetition counts: `[1, 2, 4, 8]` **per b-value**; reference SNR: `[5, 10, 20, 40]`; seed: `2026`. These are exploratory simulation choices, not a clinical protocol or fastMRI acquisition assertion. Set tissue parameters, spatial resolution, focal geometry/contrasts, and split allocation before generating the pilot.

A generated toy phantom supports initial numerical checks. Meaningful anatomy holdouts need sufficient independent anatomical models with verified use terms.

## Conventional baselines

- Average magnitude repetitions separately at each b-value, then fit ADC. With positive signals at two distinct b-values, `D = log(S(b1)/S(b2)) / (b2-b1)`. Explicitly report invalid signals, negative estimates, masks, exclusions, and constraints; avoid silent clipping.
- Fit repetition-level signals under the simulated magnitude-noise law; document noise-scale knowledge/estimation and convergence. A magnitude average does not generally retain the single-repetition noise law.
- Select published signal-domain denoising comparators compatible with the available repetitions, noise assumptions, and dimensionality. Record version, settings, adaptation, and failure cases.
- Choose the learned estimator/denoiser only after contracts, metrics, and conventional baselines are tested.

The conventional equation follows the [QIBA DWI/ADC profile](https://qibawiki.rsna.org/images/b/b0/QIBA_DWIProfile_Stage3_15Dec2022_v3.pdf), Appendix E.1. Noise-aware implementation details remain to be reviewed.

## Metrics and known-change recovery

At each budget, report signed bias, MAE/RMSE in `mm^2/s`, and regional errors relative to the noiseless acquisition-resolution reference. Declare masks. Aggregate by independent anatomy before cohort uncertainty; report exclusions/failures rather than dropping them silently.

For paired focal/unperturbed cases, define reference change `delta_ref` and estimated change `delta_hat` using the same region. Report signed change error `delta_hat-delta_ref` and recovery ratio `delta_hat/delta_ref` when `abs(delta_ref) > epsilon` for prespecified `epsilon > 0`, including both increased and decreased ADC. Handle near-zero cases separately. Distinguish assigned tissue contrast from partial-volume reference contrast; examine spillover and non-focal error. Specify paired or independent noise between cases and assess sensitivity.

Report coverage, interval width, calibration, and risk-versus-deferral curves by SNR, budget, focal size/contrast, and held-out regime. Average coverage may conceal local misses. Define false confidence as predicted bound below tolerance with realized error above tolerance; report rates over all cases and among accepted/stopped cases. Also report confident stopping with unacceptable focal-change attenuation.

## Adaptive stopping and savings

Prespecify the ADC-error quantity/spatial aggregation, tolerance `tau`, and target error risk `alpha`, or declare an exploratory sweep. Synthetic truth/masks can define evaluation but cannot enter test-time stopping. Global mean error must not be the sole local-fidelity safeguard.

Fixed-budget calibration does not establish coverage after repeated checks and adaptive stopping. Calibrate and test the entire policy on disjoint anatomies, using a justified sequential strategy or a simultaneous-over-budgets bound. Distinguish empirical validation from theoretical guarantees.

Compare repetitions saved at matched measured error risk and tolerance with the same available acquisitions. Record per-b-value counts, total counts, and ordering. Include maximum-budget non-stoppers and failures. Report uncertainty on savings with quantitative failures. Acquisition counts do not establish prospective scanner-time savings.

## Integrity and review gates

Split by source anatomy before generating variants; reserve separate training, validation, calibration, and test anatomies. Define unfamiliar acquisition/perturbation regimes before final evaluation. Fit learned transforms only on training data. Keep private manifests and frozen evaluation configuration.

Before implementation, review parameter ranges, tensor axes, invalid-signal policies, reference fit, stopping-error aggregation, tolerance/risk sweep, and sequential calibration. Minimum scientific tests are listed in `tests/README.md`. Numerical correctness alone will not validate simulation realism.
