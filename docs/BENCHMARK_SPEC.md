# Controlled simulation and quantitative evaluation

Status: CPU simulation/conventional-baseline pilot implemented and independently reviewed, 2026-10-02. Broader denoising, uncertainty, unfamiliar-regime evaluation, and stopping protocol remain proposed. `configs/pilot.json` is executable; the broader example configuration remains incomplete. Results are recorded separately in `PILOT_RESULTS.md`.

## Simulation and reference

1. Assign fine-resolution tissue fractions, `S0`, and `D`; create paired unperturbed and focal-abnormality cases. Record geometry/size and ADC contrast in physical units.
2. Generate tissue signals `S_t(b) = S0_t exp(-b D_t)`. Mix tissue signals by fractions, then apply the documented acquisition point-spread/resampling operator to signals, including boundary handling.
3. Fit the acquisition-resolution reference ADC from noiseless acquisition-resolution signals. Mixtures need not be monoexponential; this reference depends on b-values and fit. Keep assigned fine-resolution parameters separately.
4. Initially add independent Gaussian real/imaginary noise with standard deviation `sigma`, then form magnitude. Document this single-coil image-domain approximation; multicoil/k-space extensions need explicit reconstruction and covariance assumptions.
5. Define reference SNR as fixed generator-reference `S0 / sigma` for one complex component. The example's reference `S0=1` is an amplitude convention for generation/noise specification; tissue amplitudes may differ. Estimator normalization is separately defined using observed inputs, never per-voxel division by noiseless truth. Preserve individual repetitions. Do not substitute Gaussian noise on an ADC map.

The initial isotropic model does not represent all tissue diffusion or scanner artifacts. Distinguish magnitude averaging from complex averaging; comparators must follow their declared processing order.

## Reversible pilot settings

Proposed illustrative b-values: `[0, 1000] s/mm^2`; repetition counts: `[1, 2, 4, 8]` **per b-value**; reference SNR: `[5, 10, 20, 40]`; seed: `2026`. These are exploratory simulation choices, not a clinical protocol or fastMRI acquisition assertion. Set tissue parameters, spatial resolution, focal geometry/contrasts, and split allocation before generating the pilot.

A generated toy phantom supports initial numerical checks. The executable pilot selects SNR `[5,20]`, six original 16×16 geometries (train/validation/calibration/test counts `1/1/2/2`), 0.5 mm fine spacing, and 2×2 box averaging to 1 mm acquisition spacing. Synthetic ADC values are 0.0009/0.0016 mm²/s and S0 values 0.8/1.0. Circle radii are 0.5/1.5 mm; assigned focal contrasts are ±0.0003 mm²/s. These numerical choices are declared assumptions, not literature-derived biological ranges. Only test geometries are evaluated; the other split roles reserve future work and imply no training or calibration has occurred.

Coarse tissue/focal evaluation masks include every voxel with a positive fine-resolution fraction. This includes boundary partial volume. Record discrete focal area, coarse focal fraction, and reference contrast separately from nominal radius and assigned contrast. The two paired pools use identical complex noise by default, with independent pairing available as an explicitly configured sensitivity run. Budgets use prefixes of one maximum-budget pool per case. Meaningful anatomy holdouts need sufficient independent anatomical models with verified use terms.

## Conventional baselines

- Average magnitude repetitions separately at each b-value, then fit ADC. With positive signals at two distinct b-values, `D = log(S(b1)/S(b2)) / (b2-b1)`. Explicitly report invalid signals, negative estimates, masks, exclusions, and constraints; avoid silent clipping.
- Fit repetition-level signals under the simulated magnitude-noise law; document noise-scale knowledge/estimation and convergence. A magnitude average does not generally retain the single-repetition noise law.
- Select published signal-domain denoising comparators compatible with the available repetitions, noise assumptions, and dimensionality. Record version, settings, adaptation, and failure cases.
- Choose the learned estimator/denoiser only after contracts, metrics, and conventional baselines are tested.

The conventional equation follows the [QIBA DWI/ADC profile](https://qibawiki.rsna.org/images/b/b0/QIBA_DWIProfile_Stage3_15Dec2022_v3.pdf), Appendix E.1. The implemented noise-aware comparator fits nonnegative S0/ADC from individual magnitudes, with known scalar simulated sigma, using the Rice likelihood and scaled Bessel functions. See the primary [SciPy Rice distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rice.html) and [scaled I0](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.i0e.html) documentation. Inputs determine initialization and amplitude normalization; reference maps and masks enter evaluation only.

For two b-values, if the high-b independent Rice amplitude MLE is zero (`mean(magnitude²) <= 2 sigma²`), the ADC optimum is unbounded or unidentified. Return invalid/NaN ADC and preserve numerical convergence as a separate diagnostic. A finite optimizer return alone is insufficient. The reusable fitter's multi-b boundary check is sufficient rather than a general identifiability guarantee; the executable pilot requires exactly two b-values. Log-linear fits retain negative estimates and report nonpositive/nonfinite inputs explicitly.

A zero-amplitude constrained optimum also leaves ADC unidentified. Nonpositive ordered prefixes of the moment scores `mean(magnitude²)/2 - sigma²` certify that null fit using the Bessel bound `log I0(z) <= z²/4` and Abel summation. For two b-values this zero-amplitude certificate is exact; for more b-values it is sufficient. A fitted-D conditional moment check additionally rejects candidates whose conditional amplitude optimum is zero. Report certified null fits as S0=0, invalid ADC, and successful analytic convergence. No arbitrary amplitude floor converts a tiny numerical S0 into an identified ADC.

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
