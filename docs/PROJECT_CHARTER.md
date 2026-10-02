# Project charter

Status: proposed protocol, 2026-10-02. The objective is defined by the updated brief; numerical settings remain reversible until review.

## Objective and contribution

Estimate ADC maps from diffusion-weighted signals using fewer repeated acquisitions. Determine when SNR, focal-abnormality size and ADC contrast, and acquisition budget cause suppression of a real focal change, and whether uncertainty detects that failure.

The intended contribution is a reusable open benchmark with known simulation truth, quantitative failure boundaries, calibrated error-risk evaluation, and acquisition-budget comparisons. Novelty remains to be established. A negative result is scientifically useful.

## Evidence design

**Stage 1:** controlled simulation without patient data. Begin with a generated geometric phantom for numerical checks, then anatomical models such as BrainWeb after verifying use terms. Anatomy supplies structure; assigned signal and diffusion parameters are synthetic assumptions, not measured ADC. Vary focal geometry, contrast, SNR, repetitions, and acquisition settings.

**Stage 2:** optional fastMRI Prostate validation after access approval. Reserve disjoint method-input and reference repetitions. Assess agreement, repeatability, and real artifacts; a many-average patient estimate remains a noisy reference. Pixel-level lesion masks are not required for this initial extension.

## Input and target contract

- b-values: `s/mm^2`; ADC: `mm^2/s`. Record normalization, tissue parameters, resolution, noise definition, exclusions, and repetition allocation.
- Initial method input: magnitude diffusion signals with explicit b-value, repetition, and spatial axes. Direction/coil handling is an explicit later extension.
- Store fine-resolution assigned tissue parameters and acquisition-resolution references. Derive the latter from noiseless mixed/resampled diffusion signals; do not blur a ground-truth ADC map.
- Synthetic masks and paired unperturbed cases support evaluation. Do not supply test truth or oracle masks to the stopping rule.
- Uncertainty must describe a specified ADC-error quantity and spatial aggregation; a variance image alone is not a calibrated bound.

Monoexponential ADC fitting and preservation of b-values and ADC units/scaling are conventional quantitative practices. See the [QIBA DWI/ADC profile, 15 December 2022](https://qibawiki.rsna.org/images/b/b0/QIBA_DWIProfile_Stage3_15Dec2022_v3.pdf), sections 3.10, 3.12, and Appendix E. Compliance with that profile has not been assessed.

## Baselines and endpoints

Compare fixed-budget averaging followed by fitting, repetition-level fitting under an explicit noise law, relevant published denoising, and later the smallest justified learned method. Give comparators identical acquisition availability and evaluation regions. Budget each b-value explicitly.

Primary endpoints: ADC bias/error, focal-change recovery/attenuation, uncertainty coverage/calibration, selective risk, false confidence, and repetitions saved at matched error tolerance and risk. Report boundaries across SNR, size, contrast, and acquisition count. Image-quality metrics are supporting measures.

## Leakage, reproducibility, and claims

Keep every anatomy and all its perturbations, noise realizations, slices, patches, and acquisition variants in one split. Separate training, model selection, calibration, and final test anatomies; also hold out unfamiliar perturbation/acquisition regimes. Real data require participant-level grouping across sessions/scans and disjoint input/reference repetitions.

Version configuration, seed derivation, private manifests, resource checksums, software/hardware versions, exclusions, and experiment commands. Fit population transforms only on training data. Quantify reported uncertainty over independent anatomical models or participants rather than voxel counts.

Review the simulation, privacy boundaries, evaluation, and stopping-risk definition before large-scale training. Simulations support conditional absolute-bias and known-change claims; patients support retrospective agreement and robustness. Clinical validity, prospective scanner-time savings, brain-tumour performance, and biological interpretation remain outside supported claims.
